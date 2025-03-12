/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-20 21:57:08
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:07:48
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/scale_kernel.h
 * @Description: tensor原地缩放算子
 */
#ifndef SCALE_KERNEL_H
#define SCALE_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor,
                       void* stream = nullptr);
}
#endif  // SCALE_KERNEL_H