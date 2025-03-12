#ifndef EMB_KERNEL_CUH
#define EMB_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
    void embedding_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
    const tensor::Tensor& output, int32_t vocab_size, void* stream=nullptr);
}

#endif