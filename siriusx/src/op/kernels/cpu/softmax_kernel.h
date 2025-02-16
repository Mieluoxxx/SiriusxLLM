/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 22:22:17
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-16 22:51:42
 * @FilePath: /siriusx-infer/siriusx/src/op/kernels/cpu/softmax_kernel.h
 * @Description:
 */
#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif