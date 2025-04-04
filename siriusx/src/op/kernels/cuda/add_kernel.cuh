#ifndef ADD_CUH
#define ADD_CUH
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cuda(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // ADD_CUH