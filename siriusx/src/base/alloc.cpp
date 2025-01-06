/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 19:40:54
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-04 16:34:29
 * @FilePath: /SiriusX-infer/siriusx/src/base/alloc.cpp
 * @Description:
 */
#include "base/alloc.h"

namespace base {
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
}  // namespace base