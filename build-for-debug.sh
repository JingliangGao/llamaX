#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)

# create build directory
if [ ! -d "build_ponn" ]; then
    mkdir build_ponn
fi
cd build_ponn

# build project
cmake .. \
        -DCMAKE_C_COMPILER=cc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_FLAGS="" \
        -DCMAKE_C_FLAGS="" \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_INSTALL_LIBDIR=/usr/lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_AVX512=OFF \
        -DGGML_PONN=ON
make -j$(nproc)

