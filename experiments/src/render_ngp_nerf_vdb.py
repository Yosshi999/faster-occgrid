"""
Modified nerfacc/examples/train_ngp_nerf_occ.py
Original Copyright (c) 2022 Ruilong Li, UC Berkeley., MIT License.
"""

import argparse
import math
import pathlib
import time
import sqlite3
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.base import AbstractEstimator
from nerfacc.volrend import render_visibility_from_alpha, render_visibility_from_density
from nerfacc.grid import ray_aabb_intersect
import vdbgrid_cuda
from PIL import Image

def _enlarge_aabb(aabb, factor: float) -> Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])

def _meshgrid3d(res, device="cpu"):
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.long),
                torch.arange(res[1], dtype=torch.long),
                torch.arange(res[2], dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).to(device)

class VDBGridEstimator(AbstractEstimator):
    DIM: int = 3

    def __init__(self, roi_aabb, resolution=128, levels=1, **kwargs):
        super().__init__()

        if "contraction_type" in kwargs:
            raise ValueError(
                "`contraction_type` is not supported anymore for nerfacc >= 0.4.0."
            )

        # check the resolution is legal
        if isinstance(resolution, int):
            resolution = [resolution] * self.DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(resolution, Tensor), f"Invalid type: {resolution}!"
        assert resolution.shape[0] == self.DIM, f"Invalid shape: {resolution}!"

        # check the roi_aabb is legal
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(roi_aabb, Tensor), f"Invalid type: {roi_aabb}!"
        assert roi_aabb.shape[0] == self.DIM * 2, f"Invalid shape: {roi_aabb}!"

        # multiple levels of aabbs
        aabbs = torch.stack(
            [_enlarge_aabb(roi_aabb, 2**i) for i in range(levels)], dim=0
        )

        # total number of voxels
        self.cells_per_lvl = int(resolution.prod().item())
        self.levels = levels

        # Buffers
        self.register_buffer("resolution", resolution)  # [3]
        self.register_buffer("aabbs", aabbs)  # [n_aabbs, 6]
        self.register_buffer(
            "occs", torch.zeros(self.levels * self.cells_per_lvl)
        )
        self.register_buffer(
            "binaries",
            torch.zeros([levels] + resolution.tolist(), dtype=torch.bool),
        )

        # Grid coords & indices
        grid_coords = _meshgrid3d(resolution).reshape(
            self.cells_per_lvl, self.DIM
        )
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        grid_indices = torch.arange(self.cells_per_lvl)
        self.register_buffer("grid_indices", grid_indices, persistent=False)
        self.tree = vdbgrid_cuda.DensityGrid(*self.resolution, self.levels)
        self.tree.setValues(self.binaries.cpu())

    def load_state_dict(self, *args):
        super().load_state_dict(*args)
        self.tree = vdbgrid_cuda.DensityGrid(*self.resolution, self.levels)
        self.tree.setValues(self.binaries.cpu())

    @torch.no_grad()
    def sampling(
        self,
        # rays
        rays_o: Tensor,  # [n_rays, 3]
        rays_d: Tensor,  # [n_rays, 3]
        # sigma/alpha function for skipping invisible space
        sigma_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        near_plane: float = 0.0,
        far_plane: float = 1e10,
        t_min: Optional[Tensor] = None,  # [n_rays]
        t_max: Optional[Tensor] = None,  # [n_rays]
        # rendering options
        render_step_size: float = 1e-3,
        early_stop_eps: float = 1e-4,
        alpha_thre: float = 0.0,
        stratified: bool = False,
        cone_angle: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If profided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If profided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """

        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)

        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        
        t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, self.aabbs)
        t_sorted, t_indices = torch.sort(
            torch.cat([t_mins, t_maxs], dim=-1), dim=-1
        )

        ray_indices, t_starts, t_ends, chunk_indices, chunk_counts = self.tree.traverse(
            rays_o.contiguous(),
            rays_d.contiguous(),
            self.resolution,
            self.aabbs,
            t_sorted,
            t_indices,
            hits,
            near_planes,
            far_planes,
            render_step_size,
            cone_angle,
        )
        packed_info = torch.stack([chunk_indices, chunk_counts], -1)

        # skip invisible space
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
            sigma_fn is not None or alpha_fn is not None
        ):
            alpha_thre = min(alpha_thre, self.occs.mean().item())

            # Compute visibility of the samples, and filter out invisible samples
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert (
                    sigmas.shape == t_starts.shape
                ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
                masks = render_visibility_from_density(
                    t_starts=t_starts,
                    t_ends=t_ends,
                    sigmas=sigmas,
                    packed_info=packed_info,
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert (
                    alphas.shape == t_starts.shape
                ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
                masks = render_visibility_from_alpha(
                    alphas=alphas,
                    packed_info=packed_info,
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            ray_indices, t_starts, t_ends = (
                ray_indices[masks],
                t_starts[masks],
                t_ends[masks],
            )
        return ray_indices, t_starts, t_ends

    @torch.no_grad()
    def update_every_n_steps(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        n: int = 16,
    ) -> None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError(
                "You should only call this function only during training. "
                "Please call _update() directly if you want to update the "
                "field during inference."
            )
        if step % n == 0 and self.training:
            self._update(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )

    # adapted from https://github.com/kwea123/ngp_pl/blob/master/models/networks.py
    @torch.no_grad()
    def mark_invisible_cells(
        self,
        K: Tensor,
        c2w: Tensor,
        width: int,
        height: int,
        near_plane: float = 0.0,
        chunk: int = 32**3,
    ) -> None:
        """Mark the cells that aren't covered by the cameras with density -1.
        Should only be executed once before training starts.

        Args:
            K: Camera intrinsics of shape (N, 3, 3) or (1, 3, 3).
            c2w: Camera to world poses of shape (N, 3, 4) or (N, 4, 4).
            width: Image width in pixels
            height: Image height in pixels
            near_plane: Near plane distance
            chunk: The chunk size to split the cells (to avoid OOM)
        """
        assert K.dim() == 3 and K.shape[1:] == (3, 3)
        assert c2w.dim() == 3 and (
            c2w.shape[1:] == (3, 4) or c2w.shape[1:] == (4, 4)
        )
        assert K.shape[0] == c2w.shape[0] or K.shape[0] == 1

        N_cams = c2w.shape[0]
        w2c_R = c2w[:, :3, :3].transpose(2, 1)  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ c2w[:, :3, 3:]  # (N_cams, 3, 1)

        lvl_indices = self._get_all_cells()
        for lvl, indices in enumerate(lvl_indices):
            grid_coords = self.grid_coords[indices]

            for i in range(0, len(indices), chunk):
                x = grid_coords[i : i + chunk] / (self.resolution - 1)
                indices_chunk = indices[i : i + chunk]
                # voxel coordinates [0, 1]^3 -> world
                xyzs_w = (
                    self.aabbs[lvl, :3]
                    + x * (self.aabbs[lvl, 3:] - self.aabbs[lvl, :3])
                ).T
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (
                    (uvd[:, 2] >= 0)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < width)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < height)
                )
                covered_by_cam = (
                    uvd[:, 2] >= near_plane
                ) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                count = covered_by_cam.sum(0) / N_cams

                too_near_to_cam = (
                    uvd[:, 2] < near_plane
                ) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)

                cell_ids_base = lvl * self.cells_per_lvl
                self.occs[cell_ids_base + indices_chunk] = torch.where(
                    valid_mask, 0.0, -1.0
                )

    @torch.no_grad()
    def _get_all_cells(self) -> List[Tensor]:
        """Returns all cells of the grid."""
        lvl_indices = []
        for lvl in range(self.levels):
            # filter out the cells with -1 density (non-visible to any camera)
            cell_ids = lvl * self.cells_per_lvl + self.grid_indices
            indices = self.grid_indices[self.occs[cell_ids] >= 0.0]
            lvl_indices.append(indices)
        return lvl_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) -> List[Tensor]:
        """Samples both n uniform and occupied cells."""
        lvl_indices = []
        for lvl in range(self.levels):
            uniform_indices = torch.randint(
                self.cells_per_lvl, (n,), device=self.device
            )
            # filter out the cells with -1 density (non-visible to any camera)
            cell_ids = lvl * self.cells_per_lvl + uniform_indices
            uniform_indices = uniform_indices[self.occs[cell_ids] >= 0.0]
            occupied_indices = torch.nonzero(self.binaries[lvl].flatten())[:, 0]
            if n < len(occupied_indices):
                selector = torch.randint(
                    len(occupied_indices), (n,), device=self.device
                )
                occupied_indices = occupied_indices[selector]
            indices = torch.cat([uniform_indices, occupied_indices], dim=0)
            lvl_indices.append(indices)
        return lvl_indices

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        # sample cells
        if step < warmup_steps:
            lvl_indices = self._get_all_cells()
        else:
            N = self.cells_per_lvl // 4
            lvl_indices = self._sample_uniform_and_occupied_cells(N)

        for lvl, indices in enumerate(lvl_indices):
            # infer occupancy: density * step_size
            grid_coords = self.grid_coords[indices]
            x = (
                grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
            ) / self.resolution
            # voxel coordinates [0, 1]^3 -> world
            x = self.aabbs[lvl, :3] + x * (
                self.aabbs[lvl, 3:] - self.aabbs[lvl, :3]
            )
            occ = occ_eval_fn(x).squeeze(-1)
            # ema update
            cell_ids = lvl * self.cells_per_lvl + indices
            self.occs[cell_ids] = torch.maximum(
                self.occs[cell_ids] * ema_decay, occ
            )
            # suppose to use scatter max but emperically it is almost the same.
            # self.occs, _ = scatter_max(
            #     occ, indices, dim=0, out=self.occs * ema_decay
            # )
        thre = torch.clamp(self.occs[self.occs >= 0].mean(), max=occ_thre)
        self.binaries = (self.occs > thre).view(self.binaries.shape)
        self.tree = vdbgrid_cuda.DensityGrid(*self.resolution, self.levels)
        self.tree.setValues(self.binaries.cpu())



parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--db",
    type=str,
    default="./ngp_nerf_vdb.db",
    help="path to db file"
)
parser.add_argument(
    "--save_img",
    action="store_true"
)
args = parser.parse_args()

device = "cuda:0"
db_file = args.db
conn = sqlite3.connect(db_file)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS exp(data TEXT, psnr REAL, lpips REAL, traintime REAL, fps REAL, mem REAL)")
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

estimator = VDBGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)
radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

radiance_field.load_state_dict(torch.load(f"occ_models/nerf_{args.scene}.pth"))
estimator.load_state_dict(torch.load(f"occ_models/occ_{args.scene}.pth"))

# evaluation
radiance_field.eval()
estimator.eval()
if args.save_img:
    folder = pathlib.Path(f"vdb_{args.scene}")
    folder.mkdir(exist_ok=True)

psnrs = []
lpips = []
traintime = 0
rendertime = 0
with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # rendering
        tic = time.perf_counter()
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rendertime += time.perf_counter() - tic
        if args.save_img:
            Image.fromarray(rgb.mul(255).clamp(0, 255).byte().cpu().numpy()).save(str(folder / f"{i}.png"))

        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        psnrs.append(psnr.item())
        lpips.append(lpips_fn(rgb, pixels).item())
psnr_avg = sum(psnrs) / len(psnrs)
lpips_avg = sum(lpips) / len(lpips)
fps = len(test_dataset) / rendertime
mem = estimator.tree.memUsage()
print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, train time={int(traintime)}sec, fps={fps}, mem={mem}")

if not args.save_img:
    cur.executemany("INSERT INTO exp VALUES(?, ?, ?, ?, ?, ?)", [(args.scene, psnr_avg, lpips_avg, traintime, fps, mem)])
    conn.commit()
    conn.close()

