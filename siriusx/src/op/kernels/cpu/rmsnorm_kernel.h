/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-13 17:41:55
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-13 17:53:35
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/rmsnorm_kernel.h
 * @Description: 
 */
 #ifndef RMSNORM_KERNEL_H
 #define RMSNORM_KERNEL_H
 
 #include "tensor/tensor.h"
 
 namespace kernel {
 void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream = nullptr);
 }  // namespace kernel
 #endif  // RMSNORM_KERNEL_H