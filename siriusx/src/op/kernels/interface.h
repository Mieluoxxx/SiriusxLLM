/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 19:55:56
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-17 21:13:31
 * @FilePath: /siriusx-infer/siriusx/src/op/kernels/interface.h
 * @Description: 
 */
#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H

#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& in1, const tensor::Tensor& in2,
                          const tensor::Tensor& out, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);

typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t vocab_size, void* stream);

typedef void (*SwiGLUKernel)(const tensor::Tensor& in1, const tensor::Tensor& in2, const tensor::Tensor& out, void* stream);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size, 
                            const tensor::Tensor& input_q, const tensor::Tensor& input_k, const tensor::Tensor& input_pos, 
                            const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

AddKernel get_add_kernel(base::DeviceType device_type);
MatmulKernel get_matmul_kernel(base::DeviceType device_type);
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);
SwiGLUKernel get_swiglu_kernel(base::DeviceType device_type);
SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);
RoPEKernel get_rope_kernel(base::DeviceType device_type);

}  // namespace kernel
#endif  // KERNELS_INTERFACE_H