/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 19:55:56
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-07 14:36:55
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/interface.h
 * @Description:
 */
#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H

#ifndef USE_CUDA
using cudaStream_t = void*;
#else
#include <cuda_runtime.h>
#endif

#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& in1, const tensor::Tensor& in2,
                          const tensor::Tensor& out, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const CudaConfig* config);

typedef void (*MatmulKernelQuant)(const tensor::Tensor& input,
                                  const tensor::Tensor& weight,
                                  const tensor::Tensor& output,
                                  int32_t group_size,
                                  const tensor::Tensor& scale,
                                  const CudaConfig* config);

typedef void (*RMSNormKernel)(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input,
                                const tensor::Tensor& weight,
                                const tensor::Tensor& output,
                                int32_t vocab_size, void* stream);

typedef void (*SwiGLUKernel)(const tensor::Tensor& in1,
                             const tensor::Tensor& in2,
                             const tensor::Tensor& out, void* stream);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q,
                           const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos,
                           const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);

typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input,
                            void* stream);

typedef void (*ScaleSumKernel)(const tensor::Tensor& value,
                               const tensor::Tensor& scale,
                               const tensor::Tensor& output, int t, int size,
                               int stride, void* stream);

typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index,
                          int32_t seq_len, int32_t kv_dim, int32_t kv_mul,
                          int32_t head_size, const tensor::Tensor& mha_out,
                          const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor,
                          base::DeviceType device_type, CudaConfig* config);

AddKernel get_add_kernel(base::DeviceType device_type);
MatmulKernel get_matmul_kernel(base::DeviceType device_type);
MatmulKernelQuant get_matmul_quant_kernel(base::DeviceType device_type);
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);
SwiGLUKernel get_swiglu_kernel(base::DeviceType device_type);
SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);
RoPEKernel get_rope_kernel(base::DeviceType device_type);
ScaleKernel get_scale_kernel(base::DeviceType device_type);
ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);
MHAKernel get_mha_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // KERNELS_INTERFACE_H