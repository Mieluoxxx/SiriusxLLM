/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-27 20:48:37
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 17:50:17
 * @FilePath: /siriusx-infer/siriusx/include/base/cuda_config.h
 * @Description: 审查完成 0228
 */
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