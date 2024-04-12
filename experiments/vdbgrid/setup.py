from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# NOTE: OpenVDB must be built with cmake -DOPENVDB_BUILD_NANOVDB=ON -DNANOVDB_USE_INTRINSICS=ON -DNANOVDB_USE_CUDA=ON -DNANOVDB_CUDA_KEEP_PTX=ON -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" ..
setup(
    name='vdbgrid',
    ext_modules=[
        CUDAExtension('vdbgrid_cuda', [
            'csrc/grid.cpp',
            'csrc/grid_kernel.cu',
        ],
        libraries=["openvdb"],
        library_dirs=["/usr/local/lib"],
        extra_compile_args={
            "cxx": ["-O3", "-Wno-sign-compare"],
            "nvcc": ["-O3", "--extended-lambda", "--ptxas-options=-v"]
        }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    }
)
