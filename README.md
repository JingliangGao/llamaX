# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

llama.cpp is a low-cost and high-efficiency LLM inference framework  in C/C++, more details can be found [here](https://github.com/ggml-org/llama.cpp).

In this branch, we expand a new backend named "porn" to support [Glenfly](https://www.glenfly.com/) GPU. 

## Quick start
1. set up environment    
 
2. build     
```bash
chmod +x build-for-debug.sh && build-for-debug.sh
```

3. run example
```bash
./build/llama-simple -m <Your GGUF Model>
```


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/hexagon/README.md) | Snapdragon |
| [Porn [In Progress]](docs/backend/PORN.md) | Glenfly |

