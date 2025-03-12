#ifndef SWIGLU_KERNEL_H
#define SWIGLU_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& in1, const tensor::Tensor& in2,
                       const tensor::Tensor& out, void* stream);
}  // namespace kernel

#endif  // SWIGLU_KERNEL_H