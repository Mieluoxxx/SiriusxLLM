/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-10 00:10:20
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-10 03:19:23
 * @FilePath: /siriusx-infer/test/test_op/test_matmul.cpp
 * @Description:
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "../src/op/kernels/interface.h"
#include "base/base.h"

using namespace kernel;

TEST(test_matmul_cu, matmul_linear_course) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    tensor::Tensor input(base::DataType::FP32, 3, true, alloc_cpu);
    tensor::Tensor weight(base::DataType::FP32, 3, 3, true, alloc_cpu);

    input.index<float>(0) = float(1);
    input.index<float>(1) = float(1);
    input.index<float>(2) = float(-1);

    for (int i = 1; i <= 9; ++i) {
        weight.index<float>(i - 1) = float(i);
    }

    tensor::Tensor out_cpu(base::DataType::FP32, 3, true, alloc_cpu);

    kernel::get_matmul_kernel(base::DeviceType::CPU)(input, weight, out_cpu,
                                                     1.f, nullptr);
    ASSERT_EQ(out_cpu.index<float>(0), 0);
    ASSERT_EQ(out_cpu.index<float>(1), 3);
    ASSERT_EQ(out_cpu.index<float>(2), 6);
}