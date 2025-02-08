/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-08 19:16:16
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-08 19:23:05
 * @FilePath: /siriusx-infer/siriusx/src/base/alloc.cpp
 * @Description: 
 */
#include "base/alloc.h"

namespace base {
#ifdef USE_CUDA
#include <cuda_runtime.h>
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr,
                             size_t byte_size, MemcpyKind memcpy_kind,
                             void* stream, bool need_sync) const {
    // CHECK_NE: 检查两个值是否不相等，如果相等则报错
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if (!byte_size) return;

    cudaStream_t stream_ = nullptr;
    if (stream) stream_ = static_cast<CUstream_st*>(stream);

    // clang-format off
    if (memcpy_kind == MemcpyKind::CPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else if (memcpy_kind == MemcpyKind::CPU2CUDA) {
        if(!stream_) cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        else cudaMemcpyAsync (dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    } else if (memcpy_kind == MemcpyKind::CUDA2CPU) {
        if(!stream_) cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        else cudaMemcpyAsync (dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    } else if (memcpy_kind == MemcpyKind::CUDA2CUDA) {
        if(!stream_) cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        else cudaMemcpyAsync (dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
    // clang-format on

    if (need_sync) {
        cudaDeviceSynchronize();
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
    CHECK(device_type_ != DeviceType::Unknown);
    if (device_type_ == base::DeviceType::CPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }
        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }
}
#endif

#ifndef USE_CUDA
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr,
                             size_t byte_size, MemcpyKind memcpy_kind,
                             void* stream, bool need_sync) const {
    // CHECK_NE: 检查两个值是否不相等，如果相等则报错
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if (!byte_size) return;

    if (memcpy_kind == MemcpyKind::CPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
    CHECK(device_type_ != DeviceType::Unknown);
    if (device_type_ == base::DeviceType::CPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        LOG(WARNING) << "Not implemented yet.";
        std::abort();
    }
}
#endif
}  // namespace base