/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 17:27:16
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-19 20:31:56
 * @FilePath: /SiriusX-infer/siriusx/src/base/buffer.cpp
 * @Description: 
 */
#include "base/buffer.h"

#include <glog/logging.h>
#include "base/alloc.h"

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
               void* ptr, bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
    // 不具备指针所属权，Buffer不对其进行管理，不会调用接下来的资源申请流程
    if (!ptr_ && allocator_) {
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size);
    }
}

Buffer::~Buffer() {
    // 如果不是外部指针，且存在内存分配器，则释放内存
    if (!use_external_) {
        if (ptr_ && allocator_) {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}

void* Buffer::ptr() { return ptr_; }

const void* Buffer::ptr() const { return ptr_; }

size_t Buffer::byte_size() const { return byte_size_; }

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
    return allocator_;
}

void Buffer::copy_from(const Buffer& buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer.ptr_ != nullptr);

    size_t byte_size =
        byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
    const DeviceType& buffer_device = buffer.device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::Unknown &&
          current_device != DeviceType::Unknown);
    if (buffer_device == DeviceType::CPU && current_device == DeviceType::CPU) {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
    } else if(buffer_device == DeviceType::CUDA && current_device == DeviceType::CPU) {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::CUDA2CPU);
    } else if (buffer_device == DeviceType::CPU && current_device == DeviceType::CUDA) {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::CPU2CUDA);
    } else {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::CUDA2CUDA);
    }
}

void Buffer::copy_from(const Buffer* buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

    size_t dest_size = byte_size_;
    size_t src_size = buffer->byte_size_;
    size_t byte_size = src_size < dest_size ? src_size : dest_size;
    const DeviceType& buffer_device = buffer->device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::Unknown && current_device != DeviceType::Unknown);
    if (buffer_device == DeviceType::CPU && current_device == DeviceType::CPU) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
    } else if (buffer_device == DeviceType::CUDA && current_device == DeviceType::CPU) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, MemcpyKind::CUDA2CPU);
    } else if (buffer_device == DeviceType::CPU && current_device == DeviceType::CUDA) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, MemcpyKind::CPU2CUDA);
    } else {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, MemcpyKind::CUDA2CUDA);
    }
}

DeviceType Buffer::device_type() const { return device_type_; }

void Buffer::set_device_type(DeviceType device_type) {
    device_type_ = device_type;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
    return shared_from_this();
}

bool Buffer::is_external() const { return this->use_external_; }

}  // namespace base