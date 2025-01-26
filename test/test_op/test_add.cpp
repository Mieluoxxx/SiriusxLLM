/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-26 18:25:43
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-26 18:35:05
 * @FilePath: /SiriusX-infer/test/test_op/test_add.cpp
 * @Description:
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "../src/op/kernels/interface.h"
#include "base/alloc.h"
#include "base/buffer.h"

TEST(test_add_cpu, test1) {
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