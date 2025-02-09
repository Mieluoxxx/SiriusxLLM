#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H

#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& in1, const tensor::Tensor& in2,
                          const tensor::Tensor& out, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const CudaConfig* config);

AddKernel get_add_kernel(base::DeviceType device_type);
MatmulKernel get_matmul_kernel(base::DeviceType device_type);

}  // namespace kernel
#endif  // KERNELS_INTERFACE_H