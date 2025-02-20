/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-20 22:55:05
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:56:36
 * @FilePath: /siriusx-infer/test/test_op/test_scale.cpp
 * @Description: 
 */
#include "base/alloc.h"
#include "gtest/gtest.h"
#include "../src/op/kernels/interface.h"

TEST(test_sacle, test_cpu) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::FP32, size, true, alloc);
    for(int32_t i = 0; i < size; ++i) {
        t1.index<float>(i) = 2.f;
    }

    kernel::get_scale_kernel(base::DeviceType::CPU)(0.5f, t1, nullptr);
    
    for(int i = 0; i < size; i++) {
        ASSERT_EQ(t1.index<float>(1), 1.f);
    }
}