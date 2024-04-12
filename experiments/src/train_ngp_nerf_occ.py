"""
Modified nerfacc/examples/train_ngp_nerf_occ.py
Original Copyright (c) 2022 Ruilong Li, UC Berkeley., MIT License.
"""

import argparse
import math
import pathlib
import time
import sqlite3

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField
from PIL import Image

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

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
    "--save_img",
    action="store_true"
)
parser.add_argument(
    "--no_eval",
    action="store_true"
)
args = parser.parse_args()

models_path = pathlib.Path("occ_models")
models_path.mkdir(exist_ok=True)

device = "cuda:0"
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

    db_file = "./mip_nerf_occ.db"
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

    db_file = "./ngp_nerf_occ.db"

if not args.save_img:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS exp(data TEXT, psnr REAL, lpips REAL, traintime REAL, fps REAL)")


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

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in tqdm.tqdm(range(max_steps + 1)):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # render
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
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

traintime = time.time() - tic

torch.save(radiance_field.state_dict(), str(models_path / f"nerf_{args.scene}.pth"))
torch.save(estimator.state_dict(), str(models_path / f"occ_{args.scene}.pth"))
if args.no_eval:
    exit(0)

# evaluation
radiance_field.eval()
estimator.eval()
if args.save_img:
    folder = pathlib.Path(f"occ_{args.scene}")
    folder.mkdir(exist_ok=True)

psnrs = []
lpips = []
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
print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, train time={int(traintime)}sec, fps={fps}")

if not args.save_img:
    cur.executemany("INSERT INTO exp VALUES(?, ?, ?, ?, ?)", [(args.scene, psnr_avg, lpips_avg, traintime, fps)])
    conn.commit()
    conn.close()

