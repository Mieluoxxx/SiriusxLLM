#include <gtest/gtest.h>

#include "base/alloc.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "tensor/tensor.h"
#include "../src/op/kernels/interface.h"
#include <armadillo>
#include <random>

using namespace kernel;

TEST(test_rmsnorm, test_cpu) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 4;

    tensor::Tensor input(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor weight(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor output(base::DataType::FP32, size, true, alloc_cpu);

    // 填充输入张量
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < size; ++i) {
        input.index<float>(i) = input_data[i];
    }

    // 填充权重张量
    float weight_data[] = {0.5f, 1.0f, 1.5f, 2.0f};
    for (int i = 0; i < size; ++i) {
        weight.index<float>(i) = weight_data[i];
    }

    kernel::get_rmsnorm_kernel(base::DeviceType::CPU)(input, weight, output, nullptr);

    // 手动计算预期结果
    const float eps = 1e-5f;
    arma::fvec in_tensor(input_data, size, false, true);
    arma::fvec w_tensor(weight_data, size, false, true);

    // 计算均方值
    float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    float rsqrt = 1.f / std::sqrt(mean);

    // 计算预期输出
    arma::fvec expected_output = w_tensor % (rsqrt * in_tensor);

    // 验证输出张量的值
    for (int i = 0; i < size; ++i) {
        ASSERT_NEAR(output.index<float>(i), expected_output(i), 1e-5f);
    }
}

TEST(test_rmsnorm, nostream) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 15;

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

    kernel::get_rmsnorm_kernel(base::DeviceType::CUDA)(in_cuda, wei_cuda, out_cuda, nullptr);
    out_cuda.to_cpu();
    kernel::get_rmsnorm_kernel(base::DeviceType::CPU)(in_cpu, wei_cpu, out_cpu, nullptr);

    for(int i = 0; i < size; i++) {
        ASSERT_NEAR(out_cuda.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    }
}

TEST(test_rmsnorm, stream) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 15;

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

    kernel::get_rmsnorm_kernel(base::DeviceType::CUDA)(in_cuda, wei_cuda, out_cuda, stream);
    out_cuda.to_cpu();
    kernel::get_rmsnorm_kernel(base::DeviceType::CPU)(in_cpu, wei_cpu, out_cpu, nullptr);

    for(int i = 0; i < size; i++) {
        ASSERT_NEAR(out_cuda.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    }
    cudaStreamDestroy(stream);
}