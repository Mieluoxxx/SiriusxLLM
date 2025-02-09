/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-31 03:08:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-13 17:15:57
 * @FilePath: /siriusx-infer/test/test_op/test_add.cpp
 * @Description:
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "../src/op/kernels/interface.h"
#include "base/alloc.h"
#include "cuda_runtime_api.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include "../utils.cuh"
#endif

TEST(test_add, cpu_test) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    // 创建张量
    tensor::Tensor t1(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor t2(base::DataType::FP32, size, true, alloc_cpu);
    tensor::Tensor out(base::DataType::FP32, size, true, alloc_cpu);

    // 给 t1 赋值 2.0
    for (int i = 0; i < size; ++i) {
        t1.index<float>(i) = 2.f;
    }

    // 给 t2 赋值 3.0
    for (int i = 0; i < size; ++i) {
        t2.index<float>(i) = 3.f;
    }

    // 执行加法操作
    kernel::get_add_kernel(base::DeviceType::CPU)(t1, t2, out, nullptr);

    // 分配 output 数组
    float* output = new float[size];

    std::memcpy(output, out.ptr<float>(), sizeof(float) * size);

    // 验证 output 的内容是否正确
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(output[i], 5.f);  // 2.0 + 3.0 = 5.0
    }

    // 释放 output 数组
    delete[] output;
}

#ifdef USE_CUDA
// 测试CUDA设备上的加法操作
TEST(test_add, cuda_test) {
    // 获取CUDA设备分配器实例
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    // 定义张量大小
    int32_t size = 32 * 151;

    // 创建三个张量，数据类型为FP32，大小为size，使用CUDA设备分配器
    tensor::Tensor t1(base::DataType::FP32, size, true, alloc_cu);
    tensor::Tensor t2(base::DataType::FP32, size, true, alloc_cu);
    tensor::Tensor out(base::DataType::FP32, size, true, alloc_cu);

    // 设置t1张量的值为2.0
    set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
    // 设置t2张量的值为3.0
    set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

    // 调用CUDA加法核函数，将t1和t2相加，结果存储在out张量中
    kernel::get_add_kernel(base::DeviceType::CUDA)(t1, t2, out, nullptr);
    // 同步CUDA设备
    cudaDeviceSynchronize();
    // 创建一个大小为size的浮点型数组，用于存储out张量的值
    float* output = new float[size];
    // 将out张量的值从CUDA设备复制到主机内存
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float),
               cudaMemcpyDeviceToHost);
    // 遍历output数组，检查每个元素的值是否为5.0
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(output[i], 5.f);
    }

    // 释放output数组
    delete[] output;
}

TEST(test_add, nostream) {
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::FP32, size, true, alloc_cuda);
    tensor::Tensor t2(base::DataType::FP32, size, true, alloc_cuda);
    tensor::Tensor out(base::DataType::FP32, size, true, alloc_cuda);

    set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
    set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

    kernel::get_add_kernel(base::DeviceType::CUDA)(t1, t2, out, nullptr);
    cudaDeviceSynchronize();

    float* output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; ++i) {
        ASSERT_EQ(output[i], 5.f);
    }

    delete[] output;
}

TEST(test_add, stream) {
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::FP32, size, true, alloc_cuda);
    tensor::Tensor t2(base::DataType::FP32, size, true, alloc_cuda);
    tensor::Tensor out(base::DataType::FP32, size, true, alloc_cuda);

    set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
    set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_add_kernel(base::DeviceType::CUDA)(t1, t2, out, stream);
    cudaDeviceSynchronize();

    float* output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; ++i) {
        ASSERT_EQ(output[i], 5.f);
    }
    cudaStreamDestroy(stream);
    delete[] output;
}

TEST(test_add, add_align) {
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();
  
    int32_t size = 32 * 151 * 13;
  
    tensor::Tensor t1(base::DataType::FP32, size, true, alloc_cuda);
    tensor::Tensor t2(base::DataType::FP32, size, true, alloc_cuda);
    tensor::Tensor out(base::DataType::FP32, size, true, alloc_cuda);
  
    set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
    set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);
  
    kernel::get_add_kernel(base::DeviceType::CUDA)( t1, t2, out, nullptr);
    cudaDeviceSynchronize();
    float* output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(output[i], 5.4f, 0.1f);
    }
  
    delete[] output;
  }
#endif