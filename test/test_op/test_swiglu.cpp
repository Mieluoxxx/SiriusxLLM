/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 21:44:59
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-05 20:57:12
 * @FilePath: /siriusx-infer/test/test_op/test_swiglu.cpp
 * @Description: 
 */
#include <gtest/gtest.h>
#include <armadillo>
#include <random>
#include "../src/op/kernels/interface.h" // 包含 SwiGLU 核函数
#include "base/alloc.h"
#include "tensor/tensor.h"  // 包含 Tensor 类

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

TEST(test_swiglu, cpu_test) {
    // 获取 CPU 分配器
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    // 定义张量大小
    int32_t size = 4;  // 使用小规模数据便于验证

    // 创建输入张量和输出张量
    tensor::Tensor in1(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor in2(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor out(base::DataType::FP32, size, true, alloc_cpu);

    // 填充输入张量 in1
    float in1_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < size; ++i) {
        in1.index<float>(i) = in1_data[i];
    }

    // 填充输入张量 in2
    float in2_data[] = {0.5f, 1.0f, 1.5f, 2.0f};
    for (int i = 0; i < size; ++i) {
        in2.index<float>(i) = in2_data[i];
    }

    // 调用 SwiGLU 核函数
    kernel::get_swiglu_kernel(base::DeviceType::CPU)(in1, in2, out, nullptr);

    // 手动计算预期结果
    arma::fvec in1_vec(in1_data, size, false, true);
    arma::fvec in2_vec(in2_data, size, false, true);

    // 计算 Swish 激活函数
    in1_vec %= (1.0f / (1.0f + arma::exp(-in1_vec)));  // Swish(in1)
    arma::fvec expected_output = in1_vec % in2_vec;     // Swish(in1) * in2

    // 验证输出张量的值
    for (int i = 0; i < size; ++i) {
        ASSERT_NEAR(out.index<float>(i), expected_output(i), 1e-5f);
    }
}

#ifdef USE_CUDA
TEST(test_swiglu, nostream) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor in_cpu(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::FP32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for(int i = 0; i < size; i++) {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor in_cuda = in_cpu.clone();
    tensor::Tensor wei_cuda = wei_cpu.clone();
    tensor::Tensor out_cuda = out_cpu.clone();

    in_cuda.to_cuda(nullptr);
    wei_cuda.to_cuda(nullptr);
    out_cuda.to_cuda(nullptr);

    kernel::get_swiglu_kernel(base::DeviceType::CUDA)(in_cuda, wei_cuda, out_cuda, nullptr);
    out_cuda.to_cpu();
    kernel::get_swiglu_kernel(base::DeviceType::CPU)(in_cpu, wei_cpu, out_cpu, nullptr);

    for(int i = 0; i < size; i++) {
        ASSERT_NEAR(out_cuda.index<float>(i), out_cpu.index<float>(i), 1e-5);
    }
}

TEST(test_swiglu, stream) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor in_cpu(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::FP32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for(int i = 0; i < size; i++) {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
    }

    tensor::Tensor in_cuda = in_cpu.clone();
    tensor::Tensor wei_cuda = wei_cpu.clone();
    tensor::Tensor out_cuda = out_cpu.clone();

    in_cuda.to_cuda(nullptr);
    wei_cuda.to_cuda(nullptr);
    out_cuda.to_cuda(nullptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_swiglu_kernel(base::DeviceType::CUDA)(in_cuda, wei_cuda, out_cuda, stream);
    out_cuda.to_cpu();
    kernel::get_swiglu_kernel(base::DeviceType::CPU)(in_cpu, wei_cpu, out_cpu, nullptr);

    for(int i = 0; i < size; i++) {
        ASSERT_NEAR(out_cuda.index<float>(i), out_cpu.index<float>(i), 1e-5);
    }
    cudaStreamDestroy(stream);
}
#endif