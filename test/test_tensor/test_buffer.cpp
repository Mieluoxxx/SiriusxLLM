/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 17:40:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-07 16:31:30
 * @FilePath: /SiriusX-infer/test/test_tensor/test_buffer.cpp
 * @Description: 测试Buffer类
 */
#include <gtest/gtest.h>

#include "base/buffer.h"

// use_external = false, Buffer需要对内存进行管理
TEST(test_buffer, allocate) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, alloc);
    ASSERT_NE(buffer.ptr(), nullptr);
}

// use_external = true, Buffer不对内存进行管理
// 需要手动释放
TEST(test_buffer, use_external) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    float* ptr = new float[32];
    Buffer buffer(32, nullptr, ptr, true);
    ASSERT_EQ(buffer.is_external(), true);
    delete[] ptr;
}

// 查看Buffer的资源释放时机
// allocate_time, 因为曾推出局部作用于后，没有被其他引用，会在HERE1和HERE2之间释放
TEST(test_buffer, allocate_time) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    {
        Buffer buffer(32, alloc);
        ASSERT_NE(buffer.ptr(), nullptr);
        LOG(INFO) << "HERE1";
    }
    LOG(INFO) << "HERE2";
}

// allocate_time2有外部引用，所以要等整个函数执行结束后才释放
TEST(test_buffer, allocate_time2) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<Buffer> buffer;
    {
        buffer = std::make_shared<Buffer>(32, alloc);
    }
    LOG(INFO) << "HERE";
    ASSERT_NE(buffer->ptr(), nullptr);
}