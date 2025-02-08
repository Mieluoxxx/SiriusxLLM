/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-08 01:42:19
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-08 08:19:43
 * @FilePath: /siriusx-infer/test/test_tensor/test_alloc.cpp
 * @Description:
 */
#include <gtest/gtest.h>

#include <cstddef>

#include "base/alloc.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

TEST(test_alloc, AllocateAndRelease) {
    using namespace base;
    const size_t byte_size = 1024;

    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    void* ptr = alloc_cpu->allocate(byte_size);
    EXPECT_NE(ptr, nullptr);
    alloc_cpu->release(ptr);
    LOG(INFO) << "test_alloc.AllocateAndRelease passed";

#ifdef USE_CUDA
    auto alloc_cuda = CUDADeviceAllocatorFactory::get_instance();
    void* ptr_cuda = alloc_cuda->allocate(byte_size);
    EXPECT_NE(ptr_cuda, nullptr);
    alloc_cuda->release(ptr_cuda);
    LOG(INFO) << "test_alloc.AllocateAndRelease passed";
#endif
}

TEST(test_alloc, MemsetZero) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const size_t byte_size = 1024;
    char src[byte_size], dest[byte_size];

    std::memset(src, 'A', byte_size);
    std::memset(dest, 0, byte_size);

    // 调用 memset_zero 方法, 将 src 的内容设置为 0
    alloc->memset_zero(src, byte_size, nullptr, false);
    EXPECT_EQ(std::memcmp(src, dest, byte_size), 0);

#ifdef USE_CUDA
    auto alloc_cuda = CUDADeviceAllocatorFactory::get_instance();
    char src_cuda[byte_size], dest_cuda[byte_size];
    std::memset(src_cuda, 'A', byte_size);
    std::memset(dest_cuda, 0, byte_size);

    // 分配设备内存并拷贝数据
    char* d_src;
    cudaMalloc(&d_src, byte_size);
    cudaMemcpy(d_src, src, byte_size, cudaMemcpyHostToDevice);

    // 调用 memset_zero 方法，将设备内存置零
    alloc_cuda->memset_zero(d_src, byte_size, nullptr, true);

    // 将设备内存数据拷贝回主机内存并检查
    cudaMemcpy(src, d_src, byte_size, cudaMemcpyDeviceToHost);
    EXPECT_EQ(std::memcmp(src, dest, byte_size), 0);

    // 释放设备内存
    cudaFree(d_src);
#endif
}

TEST(test_alloc, CPU2CPU) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    // 创建源和目标缓冲区
    const size_t byte_size = 1024;
    char src[byte_size], dest[byte_size];

    // 初始化源缓冲区
    std::memset(src, 'A', byte_size);
    std::memset(dest, 0, byte_size);

    // 调用 memcpy 方法
    alloc->memcpy(src, dest, byte_size, MemcpyKind::CPU2CPU, nullptr, false);

    // 验证目标缓冲区的内容是否与源缓冲区一致
    EXPECT_EQ(std::memcmp(src, dest, byte_size), 0);
}

#ifdef USE_CUDA
TEST(test_alloc, CPU2CUDA) {
    using namespace base;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();
    const int byte_size = 1024;  // 1KB

    char src[byte_size];                            // 1
    char dest_cpu[byte_size];                       // 0
    memset(src, 1, byte_size);
    void* dest_cuda = alloc->allocate(byte_size);   // 0

    // src->dest_cuda, dest_cuda: 1
    alloc->memcpy(src, dest_cuda, byte_size, MemcpyKind::CPU2CUDA, nullptr, true);
    
    // dest_cuda->dest_cpu, dest_cpu: 1
    cudaError_t cuda_status = cudaMemcpy(dest_cpu, dest_cuda, byte_size, cudaMemcpyDeviceToHost);
    
    ASSERT_EQ(memcmp(src, dest_cpu, byte_size), 0);

    alloc->release(dest_cuda);
}

TEST(test_alloc, CUDA2CPU) {
    using namespace base;

    // 获取 CUDA 设备分配器实例
    auto alloc = CUDADeviceAllocatorFactory::get_instance();

    // 定义测试数据大小
    const int byte_size = 1024;  // 1KB

    // 在 CUDA 设备上分配和初始化源缓冲区
    void* src_cuda = alloc->allocate(byte_size);
    ASSERT_NE(src_cuda, nullptr) << "Failed to allocate CUDA memory";

    // 将 CUDA 源缓冲区初始化为特定值（例如 0xA5）
    cudaError_t cuda_status = cudaMemset(src_cuda, 0xA5, byte_size);
    ASSERT_EQ(cuda_status, cudaSuccess) << "CUDA memset failed: " << cudaGetErrorString(cuda_status);

    // 在 CPU 上创建目标缓冲区
    char dest_cpu[byte_size];

    // 调用 memcpy 方法，从 CUDA 拷贝到 CPU
    alloc->memcpy(src_cuda, dest_cpu, byte_size, MemcpyKind::CUDA2CPU, nullptr, true);

    // 验证目标缓冲区的内容是否与预期一致
    for (int i = 0; i < byte_size; ++i) {
        ASSERT_EQ(dest_cpu[i], static_cast<char>(0xA5)) << "Data mismatch at byte " << i;
    }

    // 释放 CUDA 内存
    alloc->release(src_cuda);
}
#endif