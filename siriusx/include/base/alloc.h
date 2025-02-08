/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 17:34:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-18 23:08:24
 * @FilePath: /SiriusX-infer/siriusx/include/base/alloc.h
 * @Description:
 */

#ifndef BASE_ALLOC_H
#define BASE_ALLOC_H

#include <glog/logging.h>

#include <map>
#include <memory>

#include "base.h"

namespace base {
enum class MemcpyKind {
    CPU2CPU = 0,
    CPU2CUDA = 1,
    CUDA2CPU = 2,
    CUDA2CUDA = 3,
};  // 内存拷贝类型

// 父类：设备资源管理器
class DeviceAllocator {
   public:
    // explicit：防止隐式转换，实际上每一个构造函数都可以加上explicit，只有在需要隐式转换时才要删去
    explicit DeviceAllocator(DeviceType device_type)
        : device_type_(device_type) {}

    virtual DeviceType device_type() const {
        return device_type_;
    }  // 返回设备类型

    // 纯虚析构函数，防止派生类忘记释放资源，const = 0
    // 表示纯虚函数，派生类必须实现
    virtual void release(void* ptr) const = 0;

    // 分配内存
    virtual void* allocate(size_t byte_size) const = 0;

    /**
     * @description: 虚内存拷贝函数，用于在设备之间拷贝数据
     * @param {void*} stc_ptr 源指针
     * @param {void*} dest_ptr 目标指针
     * @param {size_t} byte_size 字节数
     * @param {MemcpyKind} memcpy_kind 拷贝类型
     * @param {void*} stream 流
     * @param {bool} need_sync
     * 显存拷贝后进行同步，额外调用cudaDeviceSynchronize()
     */
    // const表示函数不会修改成员变量
    // 虚函数被声明为const时，任何派生类的实现也必须遵守这个约定。
    virtual void memcpy(const void* stc_ptr, void* dest_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::CPU2CPU,
                        void* stream = nullptr, bool need_sync = false) const;

    // 内存置零函数
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream,
                             bool need_sync = false);

   private:
    // 设备类型
    DeviceType device_type_ = DeviceType::Unknown;
};

// 子类：CPU设备资源管理器
class CPUDeviceAllocator : public DeviceAllocator {
   public:
    explicit CPUDeviceAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

// CPU设备资源管理器专有工厂类
class CPUDeviceAllocatorFactory {
   public:
    // 获取CPU设备资源管理器实例
    static std::shared_ptr<CPUDeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }

   private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

#ifdef USE_CUDA
// GPU设备资源管理器
struct CudaMemoryBuffer {
    void* data;
    size_t byte_size;
    bool busy;
    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
   public:
    explicit CUDADeviceAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;

   private:
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CUDADeviceAllocatorFactory {
   public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

   private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};
#endif  // USE_CUDA
}  // namespace base

#endif // SIRIUSX_BASE_ALLOC_H