#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#else
// 如果没有 CUDA，将 cudaStream_t 定义为 void*
typedef void* cudaStream_t;
#endif

namespace kernel {

struct CudaConfig {
  cudaStream_t stream = nullptr;

  ~CudaConfig() {
#ifdef USE_CUDA
    if (stream) {
      cudaStreamDestroy(stream);
    }
#endif
  }
};

}  // namespace kernel

#endif  // BLAS_HELPER_H