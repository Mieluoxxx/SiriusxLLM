#include "base/base.h"
#include "swiglu_kernel.cuh"
#include "tensor/tensor.h"

namespace kernel {
__global__ void swiglu_kernel_cuda_fp32(int size, const float* in1,
                                        const float* in2, float* out) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    extern __shared__ float shared_mem[];
    float* smem1 = shared_mem;
    float* smem2 = shared_mem + blockDim.x;

    smem1[tid] = in1[idx];
    smem2[tid] = in2[idx];
    __syncthreads();

    float value = 1.0f / (1.0f + exp(-smem1[tid]));
    smem1[tid] = smem1[tid] * value;

    out[idx] = smem1[tid] * smem2[tid];
}

void swiglu_kernel_cuda(const tensor::Tensor& in1, const tensor::Tensor& in2,
                        const tensor::Tensor& out, void* stream) {
    CHECK_EQ(in1.is_empty(), false);
    CHECK(in1.device_type() == base::DeviceType::CUDA);
    CHECK_EQ(in2.is_empty(), false);
    CHECK(in2.device_type() == base::DeviceType::CUDA);
    CHECK_EQ(out.is_empty(), false);
    CHECK(out.device_type() == base::DeviceType::CUDA);

    int size = static_cast<int32_t>(in1.size());
    int threads = 128;
    int blocks = (size + threads - 1) / threads;
    const size_t shmem = threads * sizeof(float) * 2;
    if (!stream) {
        swiglu_kernel_cuda_fp32<<<blocks, threads, shmem>>>(
            size, in1.ptr<float>(), in2.ptr<float>(), const_cast<float*>(out.ptr<float>()));
    } else {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        swiglu_kernel_cuda_fp32<<<blocks, threads, shmem, stream_>>>(
            size, in1.ptr<float>(), in2.ptr<float>(), const_cast<float*>(out.ptr<float>()));
    }
}
}  // namespace kernel