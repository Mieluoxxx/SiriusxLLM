#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH

#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
void matmul_kernel_cuda(const tensor::Tensor& input,
                        const tensor::Tensor& weight, const tensor::Tensor& output,
                        const float scale = 1.f, const CudaConfig* config = nullptr);

void matmul_kernel_cuda_qint8(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t group_size,
                              const tensor::Tensor& scale,
                              const CudaConfig* config = nullptr);
}  // namespace kernel

#endif