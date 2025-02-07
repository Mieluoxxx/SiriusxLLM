#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif  // USE_CUDA

namespace kernel {

// 定义 CudaConfig 结构体
struct CudaConfig {
#ifdef USE_CUDA
    cudaStream_t stream = nullptr;
    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
#else
    // 如果没有 USE_CUDA，提供一个空实现
    void* stream = nullptr;   // 使用 void* 模拟 cudaStream_t
    ~CudaConfig() = default;  // 默认析构函数
#endif  // USE_CUDA
};

}  // namespace kernel

#endif  // BLAS_HELPER_H