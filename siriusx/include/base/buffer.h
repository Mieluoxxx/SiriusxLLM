/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 17:18:08
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-04 17:23:26
 * @FilePath: /SiriusX-infer/siriusx/include/base/buffer.h
 * @Description: 解决内存泄漏，管理Allocator申请到的内存资源
 */
#ifndef BUFFER_H_
#define BUFFER_H_

#include <memory>

#include "base/alloc.h"

namespace base {
// NoCopyable 确保 Buffer 对象不能被复制

class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
   private:
    size_t byte_size_ = 0;  // 内存大小
    void* ptr_ = nullptr;   // 内存地址
    bool use_external_ = false;
    // ptr_来源1: 外部直接赋值，Buffer无需管理，借用关系; use_external_ = true
    // ptr_来源2: Buffer需要对内存生命周期进行管理，自动释放，use_external_ = false
    DeviceType device_type_ = DeviceType::Unknown;
    std::shared_ptr<DeviceAllocator> allocator_; // Buffer对应类别的内存分配器

   public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size,
                    std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);

    virtual ~Buffer();

    bool allocate();

    void copy_from(const Buffer& buffer) const;

    void copy_from(const Buffer* buffer) const;

    void* ptr();

    const void* ptr() const;

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type);

    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;
};
}  // namespace base

#endif