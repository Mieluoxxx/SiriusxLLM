/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 17:40:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-19 23:35:26
 * @FilePath: /SiriusX-infer/test/test_tensor/test_buffer.cpp
 * @Description: 
 */
#include <gtest/gtest.h>

#include "base/buffer.h"

// use_external = false, Buffer需要对内存进行管理
// 测试 Buffer 在管理内存时的分配功能
TEST(test_buffer, allocate) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, alloc);
    
    // ASSERT_NE: 断言两个值不相等
    ASSERT_NE(buffer.ptr(), nullptr);
}

// use_external = true, Buffer不对内存进行管理
// 需要手动释放
// 测试 Buffer 在不管理外部内存时的行为
TEST(test_buffer, use_external) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    float* ptr = new float[32];
    Buffer buffer(32, nullptr, ptr, true);

    ASSERT_EQ(buffer.is_external(), true);
    delete[] ptr;
}

// 查看Buffer的资源释放时机
// allocate_time,
// 因为曾推出局部作用于后，没有被其他引用，会在HERE1和HERE2之间释放 验证 Buffer
// 在离开作用域时自动释放内存
TEST(test_buffer, allocate_time) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    {
        Buffer buffer(32, alloc);
        ASSERT_NE(buffer.ptr(), nullptr);
        LOG(INFO) << "HERE1";
    }  // buffer释放
    LOG(INFO) << "HERE2";
}

// allocate_time2有外部引用，所以要等整个函数执行结束后才释放
// 验证 Buffer 在通过 std::shared_ptr 管理时的内存释放时机
TEST(test_buffer, allocate_time2) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<Buffer> buffer;
    {
        buffer = std::make_shared<Buffer>(32, alloc);
        // shared_ptr引用计数打印
        LOG(INFO) << "HERE: " << buffer.use_count();
    }  // buffer释放
    LOG(INFO) << "HERE";
    
    ASSERT_NE(buffer->ptr(), nullptr);
}

TEST(test_buffer, MemoryZero) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

    // 创建源和目标缓冲区
    const size_t byte_size = 1024;
    char src[byte_size];
    char dest[byte_size];

    // 初始化源缓冲区
    std::memset(src, 'A', byte_size);
    std::memset(dest, 0, byte_size);

    // 调用 memcpy 方法
    alloc->memset_zero(src, byte_size, nullptr, false);

    // 验证目标缓冲区的内容是否与源缓冲区一致
    EXPECT_EQ(std::memcmp(src, dest, byte_size), 0);
}

TEST(test_buffer, MemcpyCPU2CPU) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

    // 创建源和目标缓冲区
    const size_t byte_size = 1024;
    char src[byte_size];
    char dest[byte_size];

    // 初始化源缓冲区
    std::memset(src, 'A', byte_size);
    std::memset(dest, 0, byte_size);

    // 调用 memcpy 方法
    alloc->memcpy(src, dest, byte_size, MemcpyKind::CPU2CPU, nullptr, false);

    // 验证目标缓冲区的内容是否与源缓冲区一致
    EXPECT_EQ(std::memcmp(src, dest, byte_size), 0);
}