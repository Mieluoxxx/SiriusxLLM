/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-26 15:34:11
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-26 18:04:34
 * @FilePath: /SiriusX-infer/siriusx/src/op/kernels/interface.h
 * @Description: 
 */
#ifndef KERNELS_INTERFACE_H_
#define KERNELS_INTERFACE_H_

#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& in1, const tensor::Tensor& in2,
                          const tensor::Tensor& out, void* stream);

AddKernel get_add_kernel(base::DeviceType device_type);

}  // namespace kernel
#endif  // KERNELS_INTERFACE_H_