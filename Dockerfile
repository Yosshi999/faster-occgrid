FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

RUN apt-get update \
 && apt-get install -y libboost-iostreams-dev libtbb-dev libblosc-dev vim git tmux \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# Compute Capability. 80 for A100
ARG TCNN_CUDA_ARCHITECTURES=80
RUN pip install ninja "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

COPY openvdb /workspace/openvdb
COPY nerfacc /workspace/nerfacc
# RUN pip install /workspace/nerfacc

RUN cd openvdb \
 && mkdir build \
 && cd build \
 && cmake -DOPENVDB_BUILD_NANOVDB=ON -DNANOVDB_USE_INTRINSICS=ON -DNANOVDB_USE_CUDA=ON -DNANOVDB_CUDA_KEEP_PTX=ON -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .. \
 && make -j8 \
 && make install

RUN pip install lpips imageio pandas tabulate
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Compute Capability. 8.0 for A100
ENV TORCH_CUDA_ARCH_LIST="8.0"
RUN pip install -e /workspace/nerfacc
