/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-03-10 14:22:35
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-10 14:22:35
 * @FilePath: /SiriusxLLM/test/test_op/test_cuda.cpp
 * @Description: 测试RMSNORM CUDA优化版本的性能提升
 */
#include <gtest/gtest.h>

#include <armadillo>
#include <chrono>
#include <random>

#include "base/alloc.h"
#include "tensor/tensor.h"


#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

// v0: 基础版本实现
static __global__ void rmsnorm_f32_v0(const float* in, const float* wei,
                                      float* out, const int dim,
                                      const float eps) {
    // 每个区块只处理一个向量的归一化
    // 一个简单的实现，没有任何优化
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // 第一步：计算平方和 s = sum(x_i^2)
    float sum_squared = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        const float xi = in[i];
        sum_squared += xi * xi;
    }

    // 使用共享内存归约计算总和
    __shared__ float s_sum[1024];  // 假设最大线程数为1024
    s_sum[tid] = sum_squared;
    __syncthreads();

    // 简单的归约，没有warp优化
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    // 计算RMS并归一化
    // mean = s/d, scale = 1/sqrt(mean + ε)
    const float mean = s_sum[0] / static_cast<float>(dim);
    const float scale = rsqrtf(mean + eps);

    // 应用归一化和权重: y_i = (x_i / rms) * w_i
    for (int i = tid; i < dim; i += block_size) {
        out[i] = scale * in[i] * wei[i];
    }
}

// v1: 使用warp级别优化
#define WARP_SIZE 32
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __global__ void rmsnorm_f32_v1(const float* in, const float* wei,
                                      float* out, const int dim,
                                      const float eps) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // 计算平方和
    float sum_squared = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float xi = in[i];
        sum_squared += xi * xi;
    }

    // warp内部归约
    sum_squared = warp_reduce_sum(sum_squared);

    // 使用共享内存在warp之间归约
    __shared__ float warp_sums[32];  // 假设最多32个warp
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_squared;
    }
    __syncthreads();

    // 第一个warp负责最终归约
    if (warp_id == 0) {
        sum_squared = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
        sum_squared = warp_reduce_sum(sum_squared);
        if (lane_id == 0) {
            warp_sums[0] = sum_squared;
        }
    }
    __syncthreads();

    // 计算RMS并归一化
    const float mean = warp_sums[0] / static_cast<float>(dim);
    const float scale = rsqrtf(mean + eps);

    // 应用归一化和权重
    for (int i = tid; i < dim; i += blockDim.x) {
        out[i] = scale * in[i] * wei[i];
    }
}

// v2: 使用CUB库的WarpReduce
static __global__ void rmsnorm_f32_v2(const float* in, const float* wei,
                                      float* out, const int size,
                                      const float eps) {
    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;

    float sum = 0.0f;
    for (int i = lane_id; i < size; i += warpSize) {
        sum += in[i] * in[i];
    }

    using WarpReduce = cub::WarpReduce<float, 32>;
    __shared__ typename WarpReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = WarpReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) shared_val = sum;
    __syncthreads();
    sum = shared_val;

    float scale = rsqrtf(sum / size + eps);
    for (int i = tid; i < size; i += blockDim.x) {
        out[i] = scale * in[i] * wei[i];
    }
}

// v3: 使用float4向量化和CUB库的BlockReduce
template <int32_t BLOCK_DIM>
static __global__ void rmsnorm_f32_v3(float* in, float* wei, float* out,
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
        // 读取一个float4数据
        float4 in_float4 = in_pack[i];
        // 计算平方和并累加到sum中
        sum += in_float4.x * in_float4.x + in_float4.y * in_float4.y +
               in_float4.z * in_float4.z + in_float4.w * in_float4.w;
    }

    // 处理剩余的元素（未被向量化处理的部分）
    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        sum += in[i] * in[i];  // 计算平方和并累加到sum中
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

    // 计算均值和缩放因子
    float mean = sum / size;           // 计算均值
    float scale = rsqrtf(mean + eps);  // 计算缩放因子

    // 将输入和权重数据指针转换为float4类型，以便进行向量化处理
    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);

    // 遍历每个线程处理的float4数据，进行RMSNorm计算
    for (int i = tid; i < pack_num; i += blockDim.x) {
        // 读取输入和权重的float4数据
        float4 in_float4 = in_pack[i];
        float4 wei_float4 = wei_pack[i];
        // 计算RMSNorm后的输出数据，并写入输出数组
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

// 包装函数，用于调用CUDA kernel
void rmsnorm_kernel_cuda_v0(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            void* stream = nullptr) {
    // 检查输入、权重和输出张量是否为空
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    // 检查输入、权重和输出张量是否位于CUDA设备上
    CHECK(input.device_type() == base::DeviceType::CUDA &&
          weight.device_type() == base::DeviceType::CUDA &&
          output.device_type() == base::DeviceType::CUDA);

    // 获取输入张量的大小
    int32_t size = static_cast<int32_t>(input.size());
    // 定义一个常量eps，用于数值稳定性
    constexpr float eps = 1e-5f;

    // 获取输入、权重和输出张量的数据指针
    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    // 定义一个常量threads_num，表示线程块中的线程数量
    constexpr int threads_num = 128;
    // 如果stream不为空，则使用指定的CUDA流进行计算
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        // 调用rmsnorm_f32_v0函数进行计算，指定CUDA流
        rmsnorm_f32_v0<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                       size, eps);
    } else {
        // 否则使用默认的CUDA流进行计算
        rmsnorm_f32_v0<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

void rmsnorm_kernel_cuda_v1(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            void* stream = nullptr) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::CUDA &&
          weight.device_type() == base::DeviceType::CUDA &&
          output.device_type() == base::DeviceType::CUDA);

    int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-5f;

    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    constexpr int threads_num = 128;
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        rmsnorm_f32_v1<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                       size, eps);
    } else {
        rmsnorm_f32_v1<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

void rmsnorm_kernel_cuda_v2(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            void* stream = nullptr) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::CUDA &&
          weight.device_type() == base::DeviceType::CUDA &&
          output.device_type() == base::DeviceType::CUDA);

    int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-5f;

    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    constexpr int threads_num = 128;
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        rmsnorm_f32_v2<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                       size, eps);
    } else {
        rmsnorm_f32_v2<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

void rmsnorm_kernel_cuda_v3(const tensor::Tensor& input,
                            const tensor::Tensor& weight,
                            const tensor::Tensor& output,
                            void* stream = nullptr) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::CUDA &&
          weight.device_type() == base::DeviceType::CUDA &&
          output.device_type() == base::DeviceType::CUDA);

    int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-5f;

    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    constexpr int threads_num = 128;
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        rmsnorm_f32_v3<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr,
                                                            out_ptr, size, eps);
    } else {
        rmsnorm_f32_v3<128>
            <<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

// 计算两个张量的最大相对误差
float mre(const tensor::Tensor& a, const tensor::Tensor& b) {
    int32_t size = static_cast<int32_t>(a.size());
    const float* a_ptr = a.ptr<float>();
    const float* b_ptr = b.ptr<float>();

    float max_error = 0.0f;
    for (int32_t i = 0; i < size; ++i) {
        if (std::abs(a_ptr[i]) > 1e-6f) {
            float rel_error = std::abs((a_ptr[i] - b_ptr[i]) / a_ptr[i]);
            max_error = std::max(max_error, rel_error);
        }
    }
    return max_error;
}

// 测试不同版本的RMSNORM实现的正确性
TEST(test_rmsnorm_cuda, correctness) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    const int sizes[] = {128, 1024, 4096};

    for (int size : sizes) {
        // 创建CPU张量
        tensor::Tensor input_cpu(base::DataType::FP32, size, true, alloc_cpu);
        tensor::Tensor weight_cpu(base::DataType::FP32, size, true, alloc_cpu);
        tensor::Tensor output_cpu(base::DataType::FP32, size, true, alloc_cpu);

        // 使用随机数填充输入张量
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < size; ++i) {
            input_cpu.index<float>(i) = dis(gen);
            weight_cpu.index<float>(i) = dis(gen);
        }

        // 创建CUDA张量
        tensor::Tensor input_cuda = input_cpu.clone();
        tensor::Tensor weight_cuda = weight_cpu.clone();
        tensor::Tensor output_v0 = output_cpu.clone();
        tensor::Tensor output_v1 = output_cpu.clone();
        tensor::Tensor output_v2 = output_cpu.clone();
        tensor::Tensor output_v3 = output_cpu.clone();

        input_cuda.to_cuda();
        weight_cuda.to_cuda();
        output_v0.to_cuda();
        output_v1.to_cuda();
        output_v2.to_cuda();
        output_v3.to_cuda();

        // 运行不同版本的RMSNORM
        rmsnorm_kernel_cuda_v0(input_cuda, weight_cuda, output_v0);
        rmsnorm_kernel_cuda_v1(input_cuda, weight_cuda, output_v1);
        rmsnorm_kernel_cuda_v2(input_cuda, weight_cuda, output_v2);
        rmsnorm_kernel_cuda_v3(input_cuda, weight_cuda, output_v3);

        // 将结果拷贝回CPU
        output_v0.to_cpu();
        output_v1.to_cpu();
        output_v2.to_cpu();
        output_v3.to_cpu();

        // 检查不同版本的结果是否一致
        float error_v0_v1 = mre(output_v0, output_v1);
        float error_v0_v2 = mre(output_v0, output_v2);
        float error_v0_v3 = mre(output_v0, output_v3);

        EXPECT_LT(error_v0_v1, 1e-5f) << "Size: " << size << ", v0 vs v1";
        EXPECT_LT(error_v0_v2, 1e-5f) << "Size: " << size << ", v0 vs v2";
        EXPECT_LT(error_v0_v3, 1e-5f) << "Size: " << size << ", v0 vs v3";

        LOG(INFO) << "Size: " << size;
        LOG(INFO) << "  v0 vs v1 error: " << error_v0_v1;
        LOG(INFO) << "  v0 vs v2 error: " << error_v0_v2;
        LOG(INFO) << "  v0 vs v3 error: " << error_v0_v3;
    }
}

// 测试不同版本的RMSNORM实现的性能
TEST(test_rmsnorm_cuda, performance) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    // 测试不同大小的输入
    const int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int iterations = 1000;  // 每个版本运行的迭代次数

    LOG(INFO) << "\n========== RMSNORM CUDA性能测试 (" << iterations << " 次迭代) ==========";

    for (int size : sizes) {
        // 创建CPU张量
        tensor::Tensor input_cpu(base::DataType::FP32, size, true, alloc_cpu);
        tensor::Tensor weight_cpu(base::DataType::FP32, size, true, alloc_cpu);
        tensor::Tensor output_cpu(base::DataType::FP32, size, true, alloc_cpu);

        // 使用随机数填充输入张量
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < size; ++i) {
            input_cpu.index<float>(i) = dis(gen);
            weight_cpu.index<float>(i) = dis(gen);
        }

        // 创建CUDA张量
        tensor::Tensor input_cuda = input_cpu.clone();
        tensor::Tensor weight_cuda = weight_cpu.clone();
        tensor::Tensor output_cuda = output_cpu.clone();

        input_cuda.to_cuda();
        weight_cuda.to_cuda();
        output_cuda.to_cuda();

        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // 预热
        for (int i = 0; i < 10; ++i) {
            rmsnorm_kernel_cuda_v0(input_cuda, weight_cuda, output_cuda,
                                   stream);
            rmsnorm_kernel_cuda_v1(input_cuda, weight_cuda, output_cuda,
                                   stream);
            rmsnorm_kernel_cuda_v2(input_cuda, weight_cuda, output_cuda,
                                   stream);
            rmsnorm_kernel_cuda_v3(input_cuda, weight_cuda, output_cuda,
                                   stream);
        }

        // 测量v0版本的性能
        cudaDeviceSynchronize();
        auto start_v0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            rmsnorm_kernel_cuda_v0(input_cuda, weight_cuda, output_cuda,
                                   stream);
        }
        cudaDeviceSynchronize();
        auto end_v0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_v0 =
            end_v0 - start_v0;

        // 测量v1版本的性能
        cudaDeviceSynchronize();
        auto start_v1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            rmsnorm_kernel_cuda_v1(input_cuda, weight_cuda, output_cuda,
                                   stream);
        }
        cudaDeviceSynchronize();
        auto end_v1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_v1 =
            end_v1 - start_v1;

        // 测量v2版本的性能
        cudaDeviceSynchronize();
        auto start_v2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            rmsnorm_kernel_cuda_v2(input_cuda, weight_cuda, output_cuda,
                                   stream);
        }
        cudaDeviceSynchronize();
        auto end_v2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_v2 =
            end_v2 - start_v2;

        // 测量v3版本的性能
        cudaDeviceSynchronize();
        auto start_v3 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            rmsnorm_kernel_cuda_v3(input_cuda, weight_cuda, output_cuda,
                                   stream);
        }
        cudaDeviceSynchronize();
        auto end_v3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_v3 =
            end_v3 - start_v3;

        // 计算每次迭代的平均时间
        double time_v0 = elapsed_v0.count() / iterations;
        double time_v1 = elapsed_v1.count() / iterations;
        double time_v2 = elapsed_v2.count() / iterations;
        double time_v3 = elapsed_v3.count() / iterations;

        // 计算加速比
        double speedup_v1 = time_v0 / time_v1;
        double speedup_v2 = time_v0 / time_v2;
        double speedup_v3 = time_v0 / time_v3;

        // 使用多行格式输出结果
        LOG(INFO) << "向量大小: " << size;
        LOG(INFO) << "  v0 耗时: " << time_v0 << " ms";
        LOG(INFO) << "  v1 耗时: " << time_v1 << " ms (加速比: " << speedup_v1 << "x)";
        LOG(INFO) << "  v2 耗时: " << time_v2 << " ms (加速比: " << speedup_v2 << "x)";
        LOG(INFO) << "  v3 耗时: " << time_v3 << " ms (加速比: " << speedup_v3 << "x)";
        LOG(INFO) << "----------------------------------------";

        cudaStreamDestroy(stream);
    }
    
    LOG(INFO) << "========== 测试完成 ==========";
}