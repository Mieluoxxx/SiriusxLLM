/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-09 02:18:19
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-10 00:09:08
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/matmul_kernel.h
 * @Description:
 */
#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input,
                       const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const CudaConfig* config = nullptr);
}  // namespace kernel
#endif