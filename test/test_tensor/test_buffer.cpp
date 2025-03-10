/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 17:40:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-19 23:35:26
 * @FilePath: /SiriusxLLM/test/test_tensor/test_buffer.cpp
 * @Description:
 */
#include <gtest/gtest.h>

#include "base/buffer.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "../utils.cuh"
#endif

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

#ifdef USE_CUDA
TEST(test_buffer, CPU2CUDA) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cuda = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32;
    float* ptr = new float[size];
    for (int i = 0; i < size; i++) {
        ptr[i] = float(i);
    }
    Buffer buffer(size * sizeof(float), nullptr, ptr, true);
    buffer.set_device_type(DeviceType::CPU);
    ASSERT_EQ(buffer.is_external(), true);

    // cpu to cuda
    Buffer cu_buffer(size * sizeof(float), alloc_cuda);
    cu_buffer.copy_from(buffer);

    float* ptr2 = new float[size];
    cudaMemcpy(ptr2, cu_buffer.ptr(), size * sizeof(float),
               cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i], ptr2[i]);
    }

    delete[] ptr;
    delete[] ptr2;
}

TEST(test_buffer, CUDA2CPU) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  
    int32_t size = 32;
    Buffer cu_buffer1(size * sizeof(float), alloc_cu);
    Buffer cu_buffer2(size * sizeof(float), alloc);
    ASSERT_EQ(cu_buffer1.device_type(), DeviceType::CUDA);
    ASSERT_EQ(cu_buffer2.device_type(), DeviceType::CPU);
  
    // cu to cpu
    set_value_cu((float*)cu_buffer1.ptr(), size);
    cu_buffer2.copy_from(cu_buffer1);
  
    float* ptr2 = (float*)cu_buffer2.ptr();
    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(ptr2[i], 1.f);
    }
  }

TEST(test_buffer, CUDA2CUDA) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  
    int32_t size = 32;
    Buffer cu_buffer1(size * sizeof(float), alloc_cu);
    Buffer cu_buffer2(size * sizeof(float), alloc_cu);
  
    set_value_cu((float*)cu_buffer2.ptr(), size);
    // cu to cu
    ASSERT_EQ(cu_buffer1.device_type(), DeviceType::CUDA);
    ASSERT_EQ(cu_buffer2.device_type(), DeviceType::CUDA);
  
    cu_buffer1.copy_from(cu_buffer2);
  
    float* ptr2 = new float[size];
    cudaMemcpy(ptr2, cu_buffer1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
      ASSERT_EQ(ptr2[i], 1.f);
    }
    delete[] ptr2;
  }
#endif // USE_CUDA