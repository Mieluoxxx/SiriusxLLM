/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 17:05:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-15 20:45:11
 * @FilePath: /SiriusX-infer/siriusx/src/base/alloc_cpu.cpp
 * @Description:
 */
#include <glog/logging.h>


#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define SIRIUSX_HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::CPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (!byte_size) return nullptr;
#ifdef SIRIUSX_HAVE_POSIX_MEMALIGN
    const size_t alignment =
        (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
    int status = posix_memalign(
        (void**)&data,
        ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)), byte_size);
    if (status != 0) {
        return nullptr;
    }
    return data;
#else
    void* data = malloc(byte_size);
    return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr) {
        free(ptr);
    }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base