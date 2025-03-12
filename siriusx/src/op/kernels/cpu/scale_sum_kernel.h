/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-20 22:05:15
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:07:24
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/scale_sum_kernel.h
 * @Description: tensor加权求和算子
 */
#ifndef SCALE_SUM_KERNEL_H
#define SCALE_SUM_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void scale_sum_kernel_cpu(const tensor::Tensor& value,
                          const tensor::Tensor& scale,
                          const tensor::Tensor& output, int t, int d,
                          int stride, void* stream = nullptr);
}  // namespace kernel
#endif // SCALE_SUM_KERNEL_H