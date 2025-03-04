#include <cub/block/block_reduce.cuh>

#include "rmsnorm_kernel.cuh"

namespace kernel {

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm(float* in, float* wei, float* out,
                                        int size, float eps) {
    // 获取当前线程的索引
    const int tid = threadIdx.x;

    // 定义每个线程处理的元素数量，使用float4类型进行向量化处理
    constexpr int pack_size = 4;
    // 计算每个线程需要处理的float4数量
    const int pack_num = size / pack_size;
    // 计算向量化处理后的剩余元素数量
    const int pack_off = pack_size * pack_num;

    // 初始化sum为0，用于累加平方和
    float sum = 0.f;
    // 将输入数据指针转换为float4类型，以便进行向量化处理
    float4* in_pack = reinterpret_cast<float4*>(in);
    // 遍历每个线程处理的float4数据
    for (int i = tid; i < pack_num; i += blockDim.x) {
        // 获取当前线程处理的float4数据
        float4 in_float4 = *(in_pack + i);
        // 累加每个元素的平方和
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }

    // 遍历每个线程处理的剩余元素数量（未被向量化处理的部分）
    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        sum += in[i] * in[i];  // 累加剩余元素的平方和
    }

    // 使用CUB库中的BlockReduce进行并行归约操作，计算所有线程的sum总和
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    // 定义共享内存中的临时存储空间
    __shared__ typename BlockReduce::TempStorage temp;
    // 定义共享变量，用于存储归约后的sum值
    __shared__ float shared_val;
    // 调用BlockReduce进行归约操作，得到所有线程的sum总和
    sum = BlockReduce(temp).Sum(sum);
    // 如果当前线程是0号线程，则将归约后的sum值存入共享变量shared_val
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();   // 同步线程，确保所有线程都完成了归约操作
    sum = shared_val;  // 将共享变量shared_val赋值给sum

    // 计算RMSNorm的缩放因子scale，使用rsqrtf函数计算平方根的倒数
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    // 将权重和输出数据指针转换为float4类型，以便进行向量化处理
    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);
    // 遍历每个线程处理的float4数据，进行RMSNorm计算
    for (int i = tid; i < pack_num; i += blockDim.x) {
        // 获取当前线程处理的输入和权重float4数据
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        // 计算RMSNorm后的输出float4数据
        *(out_pack + i) = make_float4(scale * in_float4.x * wei_float4.x,
                                      scale * in_float4.y * wei_float4.y,
                                      scale * in_float4.z * wei_float4.z,
                                      scale * in_float4.w * wei_float4.w);
    }

    // 遍历每个线程处理的剩余元素数量（未被向量化处理的部分），进行RMSNorm计算
    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        out[i] = wei[i] * in[i] * scale;  // 计算RMSNorm后的输出数据
    }
}

// 定义一个名为rmsnorm_kernel_cuda的函数，用于在CUDA设备上进行RMSNorm操作
void rmsnorm_kernel_cuda(const tensor::Tensor& input,
                         const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream) {
    // 检查输入、权重和输出张量是否为空
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    // 检查输入、权重和输出张量是否位于CUDA设备上
    CHECK(input.device_type() == base::DeviceType::CUDA &&
          weight.device_type() == base::DeviceType::CUDA &&
          output.device_type() == base::DeviceType::CUDA);

    // 定义一个常量eps，用于防止除零错误
    const float eps = 1e-5f;

    // 获取输入张量的大小
    const int32_t size = static_cast<int32_t>(input.size());
    // 获取输入、权重和输出张量的指针
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    // 定义一个常量threads_num，表示线程块中的线程数量
    constexpr int threads_num = 128;
    // 如果stream不为空，则使用指定的CUDA流进行计算
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        // 调用row_rmsnorm函数进行计算，指定CUDA流
        row_rmsnorm<128><<<1, threads_num, 0, stream_>>>(
            in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
        // 否则使用默认的CUDA流进行计算
        row_rmsnorm<128>
            <<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}
}  // namespace kernel