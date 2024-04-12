#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <torch/extension.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
// #include <thrust/device_vector.h>

// #define ALG_DDA_BRANCH
// #define ALG_DDA_SKIP
// #define ALG_HDDA_BRANCH
#define ALG_HDDA_SKIP
// #define ALG_HDDA_HYBRID

inline __device__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ float _calc_dt(
    const float t, const float cone_angle,
    const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}

// Helper function for 2d-morton coding.
inline __device__ int64_t spread2(int64_t x) {
    x &=                0x00000000FFFFFFFF;
    x = (x | x << 32) & 0x0000FFFF0000FFFF;
    x = (x | x << 16) & 0x00FF00FF00FF00FF;
    x = (x | x << 8) &  0x0F0F0F0F0F0F0F0F;
    x = (x | x << 4) &  0x3333333333333333;
    x = (x | x << 2) &  0x5555555555555555;
    return x;
}

// Convert index to morton code.
inline __device__ int64_t index2key(int64_t x, int64_t y) {
    return spread2(x) | (spread2(y) << 1);
}

__global__ void indices2keys_kernel(
    const int64_t n_points,
    const int64_t *indices,  // [numel, 2]
    int64_t *out  // [numel]
) {
    CUDA_KERNEL_LOOP_TYPE(index, n_points, int64_t) {
        out[index] = index2key(indices[index*2], indices[index*2+1]);
    }
}

// march until t_mid is right after t_target
inline __device__ float advance_time(float t, float cone_angle, float step_size, float t_target) {
    float dt = _calc_dt(t, cone_angle, step_size, 1e10f);
    float advance_n = ceilf(fmaxf((t_target - t - dt * 0.5f) / dt, 0));
    return t + advance_n * dt;
}

__global__ void traverse_kernel(
    int n_grids,
    const nanovdb::MaskGrid** grids,
    int64_t n_rays,
    const float *rays_o,  // [n_rays, 3]
    const float *rays_d,  // [n_rays, 3]
    const int32_t *resolution,  // [3]
    const float *aabbs,  // [n_grids, 6]
    // sorted intersections
    const bool *hits,         // [n_rays, n_grids]
    const float *t_sorted,    // [n_rays, n_grids * 2]
    const int64_t *t_indices, // [n_rays, n_grids * 2]
    // options
    const float *near_planes,  // [n_rays]
    const float *far_planes,  // [n_rays]
    const float step_size,
    const float cone_angle,
    const bool first_pass,  // If True, only chunk_counts are filled
    int64_t *chunk_counts,  // [n_rays] the number of traversed samples
    int64_t *chunk_indices,  // [n_rays]
    int64_t *ray_indices,  // [flattened_samples]
    float *t_left,  // [flattened_samples]
    float *t_right  // [flattened_samples]
) {
    using GridT = nanovdb::MaskGrid;
    using TreeT = GridT::TreeType;

    CUDA_KERNEL_LOOP_TYPE(index, n_rays, int64_t) {
        int32_t base_hits = index * n_grids;
        int32_t base_t_sorted = index * n_grids * 2;
        int32_t end_t_sorted = base_t_sorted + n_grids * 2 - 1;

        int n_trav = 0;
        float tlast = near_planes[index];
        // loop over all intersections along the ray.
        for (int32_t i = base_t_sorted; i < end_t_sorted; i++) {
            // whether this is the entering or leaving for this level of grid.
            bool is_entering = t_indices[i] < n_grids;
            int64_t level = t_indices[i] % n_grids;

            if (!hits[base_hits + level]) {
                continue; // this grid is not hit.
            }
            if (!is_entering) {
                // we are leaving this grid. Are we inside the next grid?
                bool next_is_entering = t_indices[i + 1] < n_grids;
                if (next_is_entering) continue; // we are outside next grid.
                level = t_indices[i + 1] % n_grids;
                if (!hits[base_hits + level]) {
                    continue; // this grid is not hit.
                }
            }

            float this_tmin = fmaxf(t_sorted[i], near_planes[index]);
            float this_tmax = fminf(t_sorted[i + 1], far_planes[index]);   
            if (this_tmin >= this_tmax) continue; // this interval is invalid. e.g. (0.0f, 0.0f)        
            tlast = advance_time(tlast, cone_angle, step_size, this_tmin);

            nanovdb::Vec3f aabb_min(aabbs[level * 6], aabbs[level * 6 + 1], aabbs[level * 6 + 2]);
            nanovdb::Vec3f aabb_max(aabbs[level * 6 + 3], aabbs[level * 6 + 4], aabbs[level * 6 + 5]);
            nanovdb::Vec3f res(resolution[0], resolution[1], resolution[2]);

            auto acc = grids[level]->tree().getAccessor();
            nanovdb::Vec3f ray_o(rays_o[index * 3], rays_o[index * 3 + 1], rays_o[index * 3 + 2]);
            nanovdb::Vec3f ray_d(rays_d[index * 3], rays_d[index * 3 + 1], rays_d[index * 3 + 2]);
            // to index-space
            ray_o = (ray_o - aabb_min) / (aabb_max - aabb_min) * res;
            ray_d = ray_d / (aabb_max - aabb_min) * res;

            nanovdb::Ray<float> ray(ray_o, ray_d, this_tmin, this_tmax);

            nanovdb::TreeMarcher<TreeT::LeafNodeType, decltype(ray), decltype(acc)> marcher(acc);
            nanovdb::DDA<decltype(ray)> dda;
#ifdef ALG_DDA_BRANCH
            dda.init(ray, this_tmin, this_tmax);
            bool ddaIsValid = true;
            do {
                float dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                if (acc.getValue(dda.voxel())) {
                    while (tlast + dt / 2 < dda.next()) {
                        if (!first_pass) {
                            int64_t chunk_offset = chunk_indices[index];
                            ray_indices[chunk_offset + n_trav] = index;
                            t_left[chunk_offset + n_trav] = tlast;
                            t_right[chunk_offset + n_trav] = tlast + dt;
                        }
                        n_trav++;
                        dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                        tlast += dt;
                    }
                } else {
                    while (tlast + dt / 2 < dda.next()) tlast += dt;
                }
            } while (dda.step());
#endif
#ifdef ALG_DDA_SKIP
            dda.init(ray, this_tmin, this_tmax);
            bool ddaIsValid = true;
            do {
                do {
                    if (acc.getValue(dda.voxel())) break;
                    ddaIsValid = dda.step();
                } while (ddaIsValid);
                if (!ddaIsValid) break;
                tlast = advance_time(tlast, cone_angle, step_size, dda.time());
                float dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                while (tlast + dt / 2 < dda.next()) {
                    if (!first_pass) {
                        int64_t chunk_offset = chunk_indices[index];
                        ray_indices[chunk_offset + n_trav] = index;
                        t_left[chunk_offset + n_trav] = tlast;
                        t_right[chunk_offset + n_trav] = tlast + dt;
                    }
                    n_trav++;
                    dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                    tlast += dt;
                }
            } while (dda.step());
#endif
#ifdef ALG_HDDA_BRANCH
            float t0 = ray.t0() + 1e-6f;
            float t1 = ray.t1() - 1e-6f;
            nanovdb::HDDA<decltype(ray), nanovdb::Coord> hdda;
            bool ddaIsValid = true;
            float dt;
            if (!ray.clip(acc.root().bbox()) || t0 > t1) continue;
            auto ijk = nanovdb::RoundDown<nanovdb::Coord>(ray(t0));
            int dim = acc.getDim(ijk, ray);
            hdda.init(ray, t0, t1, dim);
            do {
                float dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                if (acc.isActive(ijk)) {
                    while (tlast + dt / 2 < hdda.next()) {
                        if (!first_pass) {
                            int64_t chunk_offset = chunk_indices[index];
                            ray_indices[chunk_offset + n_trav] = index;
                            t_left[chunk_offset + n_trav] = tlast;
                            t_right[chunk_offset + n_trav] = tlast + dt;
                        }
                        n_trav++;
                        dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                        tlast += dt;
                    }
                } else {
                    while (tlast + dt / 2 < hdda.next()) tlast += dt;
                }

                ddaIsValid = hdda.step();
                ijk = hdda.voxel();
                dim = acc.getDim(ijk, ray);
                hdda.update(ray, dim);
            } while (ddaIsValid);
#endif
#ifdef ALG_HDDA_SKIP
            float t0 = ray.t0() + 1e-6f;
            float t1 = ray.t1() - 1e-6f;
            nanovdb::HDDA<decltype(ray), nanovdb::Coord> hdda;
            bool ddaIsValid = true;
            float dt;
            if (!ray.clip(acc.root().bbox()) || t0 > t1) continue;
            auto ijk = nanovdb::RoundDown<nanovdb::Coord>(ray(t0));
            int dim = acc.getDim(ijk, ray);
            hdda.init(ray, t0, t1, dim);
            do {
                do {
                    if (acc.isActive(ijk)) break;

                    ddaIsValid = hdda.step();
                    ijk = hdda.voxel();
                    dim = acc.getDim(ijk, ray);
                    hdda.update(ray, dim);
                } while (ddaIsValid);
                if (!ddaIsValid) break;
                tlast = advance_time(tlast, cone_angle, step_size, hdda.time());
                dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                while (tlast + dt / 2 < hdda.next()) {
                    if (!first_pass) {
                        int64_t chunk_offset = chunk_indices[index];
                        ray_indices[chunk_offset + n_trav] = index;
                        t_left[chunk_offset + n_trav] = tlast;
                        t_right[chunk_offset + n_trav] = tlast + dt;
                    }
                    n_trav++;
                    dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                    tlast += dt;
                }

                ddaIsValid = hdda.step();
                ijk = hdda.voxel();
                dim = acc.getDim(ijk, ray);
                hdda.update(ray, dim);
            } while (ddaIsValid);
#endif
#ifdef ALG_HDDA_HYBRID
            const int leafdim = TreeT::LeafNodeType::dim();
            // assert(leafdim == 8);
            float t0 = ray.t0() + 1e-6f;
            float t1 = ray.t1() - 1e-6f;
            nanovdb::HDDA<decltype(ray), nanovdb::Coord> hdda;
            bool ddaIsValid = true;
            float dt;
            if (!ray.clip(acc.root().bbox()) || t0 > t1) continue;
            auto ijk = nanovdb::RoundDown<nanovdb::Coord>(ray(t0));
            int dim = acc.getDim(ijk, ray);
            // whether we are looking the interval as a tile
            // in non-tilemode (looking at a single voxel), traverse anyway as 8-dim tile
            bool tileMode = dim >= leafdim;
            hdda.init(ray, t0, t1, nanovdb::Max(leafdim, dim));
            do {
                while (ddaIsValid && tileMode) {
                    if (acc.isActive(ijk)) break;  // it is an occupied tile. traverse.

                    ddaIsValid = hdda.step();
                    ijk = hdda.voxel();
                    dim = acc.getDim(ijk, ray);
                    tileMode = dim >= leafdim;
                    hdda.update(ray, nanovdb::Max(leafdim, dim));
                }
                if (!ddaIsValid) break;
                if (hdda.time() > hdda.next()) break;
                dda.init(ray, hdda.time(), hdda.next());
                tlast = advance_time(tlast, cone_angle, step_size, hdda.time());
                do {
                    dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                    if (acc.isActive(dda.voxel())) {
                        while (tlast + dt / 2 < dda.next()) {
                            if (!first_pass) {
                                int64_t chunk_offset = chunk_indices[index];
                                ray_indices[chunk_offset + n_trav] = index;
                                t_left[chunk_offset + n_trav] = tlast;
                                t_right[chunk_offset + n_trav] = tlast + dt;
                            }
                            n_trav++;
                            dt = _calc_dt(tlast, cone_angle, step_size, 1e10f);
                            tlast += dt;
                        }
                    } else {
                        while (tlast + dt / 2 < dda.next()) tlast += dt;
                    }
                } while (dda.step());

                ddaIsValid = hdda.step();
                ijk = hdda.voxel();
                dim = acc.getDim(ijk, ray);
                tileMode = dim >= leafdim;
                hdda.update(ray, nanovdb::Max(leafdim, dim));
            } while (ddaIsValid);
#endif
        }
        if (first_pass)
            chunk_counts[index] = n_trav;
    }
}

torch::Tensor indices2keys(torch::Tensor indices) {
    indices = indices.to(torch::kLong);
    int threads = 256;
    int64_t n_points = indices.size(0);
    torch::Tensor keys = torch::empty({n_points}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    indices2keys_kernel<<<at::cuda::detail::GET_BLOCKS(n_points, threads), threads>>>(
        n_points,
        indices.data_ptr<int64_t>(),
        keys.data_ptr<int64_t>()
    );
    return keys;
}


std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> launch_traverse_kernel(
    // thrust::device_vector<const nanovdb::BoolGrid*>& grids,
    const torch::Tensor grids,
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor resolution,
    const torch::Tensor aabbs,
    const torch::Tensor t_sorted,
    const torch::Tensor t_indices,
    const torch::Tensor hits,
    const torch::Tensor near_planes,
    const torch::Tensor far_planes,
    const float step_size,
    const float cone_angle
) {
    int64_t n_rays = rays_o.size(0);
    torch::Tensor chunk_counts = torch::empty({n_rays}, rays_o.options().dtype(torch::kLong));
    int threads = 256;

    traverse_kernel<<<at::cuda::detail::GET_BLOCKS(n_rays, threads), threads>>>(
        // grids.size(),
        // thrust::raw_pointer_cast(grids.data()),
        grids.size(0),
        reinterpret_cast<const nanovdb::MaskGrid**>(grids.data_ptr<int64_t>()),
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        resolution.data_ptr<int32_t>(),
        aabbs.data_ptr<float>(),
        hits.data_ptr<bool>(),         // [n_rays, n_grids]
        t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
        t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
        near_planes.data_ptr<float>(),
        far_planes.data_ptr<float>(),
        step_size,
        cone_angle,
        true,
        chunk_counts.data_ptr<int64_t>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr);

    torch::Tensor cumsum = torch::cumsum(chunk_counts, 0, chunk_counts.scalar_type());
    int64_t n_samples = cumsum[-1].item<int64_t>();
    torch::Tensor chunk_indices = cumsum - chunk_counts;
    torch::Tensor ray_indices = torch::zeros({n_samples}, chunk_counts.options().dtype(torch::kLong));
    torch::Tensor t_left = torch::zeros({n_samples}, chunk_counts.options().dtype(torch::kFloat32));
    torch::Tensor t_right = torch::zeros({n_samples}, chunk_counts.options().dtype(torch::kFloat32));

    traverse_kernel<<<at::cuda::detail::GET_BLOCKS(n_rays, threads), threads>>>(
        // grids.size(),
        // thrust::raw_pointer_cast(grids.data()),
        grids.size(0),
        reinterpret_cast<const nanovdb::MaskGrid**>(grids.data_ptr<int64_t>()),
        n_rays,
        rays_o.data_ptr<float>(),
        rays_d.data_ptr<float>(),
        resolution.data_ptr<int32_t>(),
        aabbs.data_ptr<float>(),
        hits.data_ptr<bool>(),         // [n_rays, n_grids]
        t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
        t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
        near_planes.data_ptr<float>(),
        far_planes.data_ptr<float>(),
        step_size,
        cone_angle,
        false,
        chunk_counts.data_ptr<int64_t>(),
        chunk_indices.data_ptr<int64_t>(),
        ray_indices.data_ptr<int64_t>(),
        t_left.data_ptr<float>(),
        t_right.data_ptr<float>());

    return {ray_indices, t_left, t_right, chunk_indices, chunk_counts};
}
