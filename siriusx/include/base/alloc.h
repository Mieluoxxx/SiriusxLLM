/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 17:34:31
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-04 17:35:17
 * @FilePath: /SiriusX-infer/siriusx/include/base/alloc.h
 * @Description: 设备资源管理器
 */
#ifndef ALLOC_H_
#define ALLOC_H_

#include <glog/logging.h>

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
     * @param {bool} need_sync 显存拷贝后进行同步，额外调用cudaDeviceSynchronize()
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

// CPU设备资源管理器工厂类
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

// 设备资源管理器工厂类
class DeviceAllocatorFactory {
   public:
    // 获取设备资源管理器实例
    static std::shared_ptr<DeviceAllocator> get_instance(
        DeviceType device_type) {
        if (device_type == DeviceType::CPU) {
            return CPUDeviceAllocatorFactory::get_instance();
        } else if (device_type == DeviceType::CUDA) {
            LOG(WARNING) << "Not implemented yet.";
            return nullptr;
        } else {
            // LOG(FATAL) << "This device type of allocator is not supported!";
            return nullptr;
        }
    }
};

}  // namespace base

#endif