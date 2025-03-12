/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-03-05 09:06:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-05 12:20:19
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cuda/emb_kernel.cu
 * @Description: 
 */
#include "base/base.h"
#include "emb_kernel.cuh"

namespace kernel {
__global__ void emb_kernel_cuda_fp32(int32_t vocab_size, int32_t token_num,
                                     int32_t weight_dim,
                                     const int32_t* input_ptr,
                                     const float* weight_ptr,
                                     float* output_ptr) {
    int32_t token_idx = blockIdx.x;
    if (token_idx >= token_num) {
        return;
    }
    int32_t token = input_ptr[token_idx];
    if (token >= vocab_size) {
        return;
    }

    float* output_ptr_start = output_ptr + token_idx * weight_dim;
    const float* weight_ptr_start = weight_ptr + token * weight_dim;

    for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
        output_ptr_start[i] = weight_ptr_start[i];
    }
}

void embedding_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                     const tensor::Tensor& output, int32_t vocab_size,
                     void* stream) {
    tensor::Tensor input_cuda;
    if (input.device_type() != base::DeviceType::CUDA) {
        input_cuda = input.clone();
        input_cuda.to_cuda();
    }
    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);
    CHECK(weight.device_type() == output.device_type());
    CHECK(output.device_type() == base::DeviceType::CUDA);

    constexpr int32_t max_seq_len = 512;
    constexpr int32_t thread_num = 128;

    int32_t* in_ptr = input_cuda.ptr<int32_t>();
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        emb_kernel_cuda_fp32<<<max_seq_len, thread_num, 0, stream_>>>(
            vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    } else {
        emb_kernel_cuda_fp32<<<max_seq_len, thread_num>>>(vocab_size, input_num,
                                                          weight_dim, in_ptr,
                                                          wei_ptr, out_ptr);
    }
}
}  // namespace kernel