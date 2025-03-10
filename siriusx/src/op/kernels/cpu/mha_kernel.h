/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-20 20:29:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:45:07
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/mha_kernel.h
 * @Description: 
 */
#ifndef MHA_KERNEL_H
#define MHA_KERNEL_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
void mha_kernel_cpu(int32_t pos, int32_t head_num, int32_t layer_index,
                int32_t seq_len, int32_t kv_dim, int32_t kv_mul,
                int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config);
}  // namespace kernel

#endif  // MHA_KERNEL_H