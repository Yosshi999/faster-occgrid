#include <nanovdb/util/Ray.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <torch/extension.h>
#include <vector>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

torch::Tensor indices2keys(torch::Tensor indices);

using PackedSegments = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

PackedSegments launch_traverse_kernel(
    // thrust::device_vector<const nanovdb::BoolGrid*> grids,
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
    const float cone_angle);

using openvdb::math::Coord;


class DensityGrid {
    using GridT = openvdb::MaskGrid;
    using NanoGridT = nanovdb::MaskGrid;
    using NanoValueT = nanovdb::ValueMask;
public:
    DensityGrid(int rx, int ry, int rz, int levels)
    : mX(rx), mY(ry), mZ(rz), mLevels(levels), handle(levels) {
        openvdb::initialize();
        for (int i=0; i<levels; i++) {
            mOpenGrid.emplace_back(GridT::create());
        }
    }

    void setValues(torch::Tensor binaries) {
        TORCH_CHECK(!binaries.device().is_cuda(), "binaries must be CPU tensor");
        TORCH_CHECK(binaries.is_contiguous(), "binaries must be contiguous");
        TORCH_CHECK(binaries.size(0) == mLevels, "levels mismatch");
        mNanoGridD = torch::zeros({mLevels}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

        const bool* ptr = binaries.data_ptr<bool>();
        int sG = mX * mY * mZ;
        int sX = mY * mZ;
        int sY = mZ;
        for (int g=0; g<mLevels; g++) {
            auto accessor = mOpenGrid[g]->getAccessor();
            for (int x = 0; x < mX; x++) {
                for (int y = 0; y < mY; y++) {
                    for (int z = 0; z < mZ; z++) {
                        if (ptr[sG * g + sX * x + sY * y + z]) {
                            accessor.setValue(Coord(x, y, z), true);
                        }
                    }
                }
            }

            mOpenGrid[g]->tree().prune();

            handle[g] = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*mOpenGrid[g]);
            handle[g].deviceUpload();
            // mNanoGridH[g] = handle[g].deviceGrid<NanoValueT>();
            // TORCH_CHECK(mNanoGridH[g], "GPU handling error");
            auto nano = handle[g].deviceGrid<NanoValueT>();
            TORCH_CHECK(nano, "GPU handling error");
            mNanoGridD[g] = reinterpret_cast<int64_t>(nano);
        }
        // mNanoGridD = mNanoGridH;
    }

    PackedSegments traverse(
        torch::Tensor rays_o,  // [n_rays, 3]
        torch::Tensor rays_d,  // [n_rays, 3]
        torch::Tensor resolution,
        torch::Tensor aabbs,
        torch::Tensor t_sorted,
        torch::Tensor t_indices,
        torch::Tensor hits,
        torch::Tensor near_planes,  // [n_rays]
        torch::Tensor far_planes,  // [n_rays]
        const float step_size,
        const float cone_angle
    ) {
        CHECK_INPUT(rays_o);
        CHECK_INPUT(rays_d);
        CHECK_INPUT(near_planes);
        CHECK_INPUT(far_planes);

        return launch_traverse_kernel(
            mNanoGridD,
            rays_o,
            rays_d,
            resolution,
            aabbs,
            t_sorted,
            t_indices,
            hits,
            near_planes,
            far_planes,
            step_size,
            cone_angle
        );
    }

    uint64_t memUsage() {
        uint64_t mem = 0;
	for (int i = 0; i < mLevels; i++) {
            // mem += reinterpret_cast<const NanoGridT*>(mNanoGridD[i].item<int64_t>()) -> memUsage();
            mem += mOpenGrid[i]->memUsage();
        }
        return mem;
    }

private:
    std::vector<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> handle;
    std::vector<GridT::Ptr> mOpenGrid;
    // thrust::host_vector<const NanoGridT*> mNanoGridH;
    // thrust::device_vector<const NanoGridT*> mNanoGridD;
    torch::Tensor mNanoGridD;

    /* resolution of the grid */
    int mX, mY, mZ;
    int mLevels;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("indices2keys", &indices2keys);
    py::class_<DensityGrid>(m, "DensityGrid")
        .def(py::init<const int, const int, const int, const int>())
        .def("setValues", &DensityGrid::setValues)
        .def("traverse", &DensityGrid::traverse)
    	.def("memUsage", &DensityGrid::memUsage);
}
