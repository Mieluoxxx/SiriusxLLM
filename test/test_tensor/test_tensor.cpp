/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-17 20:21:41
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-19 16:11:00
 * @FilePath: /SiriusX-infer/test/test_tensor/test_tensor.cpp
 * @Description:
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

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