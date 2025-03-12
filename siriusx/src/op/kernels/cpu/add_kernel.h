#ifndef ADD_KERNEL_H
#define ADD_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& in1, const tensor::Tensor& in2,
                    const tensor::Tensor& out, void* stream = nullptr);
}  // namespace kernel

#endif  // ADD_KERNEL_H