/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-17 20:21:41
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-26 13:09:53
 * @FilePath: /SiriusX-infer/test/test_tensor/test_tensor.cpp
 * @Description: 
 */
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>

#include "base/alloc.h"
#include "base/base.h"
#include "base/buffer.h"
#include "tensor/tensor.h"

// 测试 Tensor 类的初始化，
TEST(test_tensor, init1) {
    using namespace base;

    // 获取 CPU 设备分配器实例
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::FP32, size, true, alloc_cpu);
    ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, init2) {
    using namespace base;
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::FP32, size, false, alloc_cpu);
    ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, init3) {
    using namespace base;
    float* ptr = new float[32];
    ptr[0] = 31;
    tensor::Tensor t1(base::DataType::FP32, 32, false, nullptr, ptr);

    ASSERT_EQ(t1.is_empty(), false);
    ASSERT_EQ(t1.ptr<float>(), ptr);
    ASSERT_EQ(*t1.ptr<float>(), 31);
}

TEST(test_tensor, assign) {
    using namespace base;
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    // 创建一个 32x32 的 Tensor
    tensor::Tensor t1_cpu(DataType::FP32, 32, 32, true, alloc_cpu);

    ASSERT_EQ(t1_cpu.is_empty(), false);

    int32_t size = 32 * 32;
    float* ptr = new float[size];
    for (int i = 0; i < size; i++) {
        ptr[i] = float(i);
    }
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr,
                                 true);  // buffer 释放的时候会释放 ptr
    buffer->set_device_type(DeviceType::CPU);

    ASSERT_EQ(t1_cpu.assign(buffer), true);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
    delete[] ptr;
}

TEST(test_tensor, clone_cpu) {
    using namespace base;
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::FP32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);

    for (int i = 0; i < 32 * 32; ++i) {
        t1_cpu.index<float>(i) = 1.f;
    }

    tensor::Tensor t2_cpu = t1_cpu.clone();
    float* p2 = new float[32 * 32];
    std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }
    std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }
    delete[] p2;
}