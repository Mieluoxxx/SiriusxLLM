#ifndef SWIGLU_KERNEL_CUH
#define SWIGLU_KERNEL_CUH
#include "tensor/tensor.h"
namespace kernel {
void swiglu_kernel_cuda(const tensor::Tensor& in1, const tensor::Tensor& in2,
                        const tensor::Tensor& out, void* stream);
}
#endif // SWIGLU_KERNEL_CUH