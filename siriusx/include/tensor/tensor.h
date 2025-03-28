/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-15 21:35:50
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-15 22:12:47
 * @FilePath: /SiriusxLLM/siriusx/include/tensor/tensor.h
 * @Description:
 */
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

#include "base/base.h"
#include "base/buffer.h"
#include "base/cuda_config.h"

namespace tensor {
class Tensor {
   public:
    // 构造函数与析构函数
    explicit Tensor() = default;
    // clang-format off
    explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    // clang-format on

    // 张量操作
    void to_cpu();
    void to_cuda(cudaStream_t stream = nullptr);
    bool is_empty() const;
    void reshape(const std::vector<int32_t>& dims);
    tensor::Tensor clone() const;

    // 内存管理
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                     base::DataType data_type, bool need_alloc, void* ptr);
    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                  bool need_realloc = false);
    bool assign(std::shared_ptr<base::Buffer> buffer);

    // ptr方法用于获取Tensor内部数据的指针
    // 支持获取整个数据区域的指针或指定偏移量处的指针。
    // 非常方便内存拷贝
    template <typename T>
    T* ptr();
    template <typename T>
    const T* ptr() const;
    template <typename T>
    T* ptr(int64_t index);
    template <typename T>
    const T* ptr(int64_t index) const;
    
    // index方法用于通过偏移量（offset）访问Tensor中的单个元素
    // 并返回该元素的引用或常量引用。
    // 非常方便检查单个元素的值
    template <typename T>
    T& index(int64_t offset);
    template <typename T>
    const T& index(int64_t offset) const;

    // 属性获取
    size_t size() const;
    size_t byte_size() const;
    int32_t dims_size() const;
    base::DataType data_type() const;
    int32_t get_dim(int32_t idx) const;
    const std::vector<int32_t>& dims() const;
    // strides方法用于获取Tensor的步长（strides），即每个维度上的元素间隔。
    // 例如{1,2,3,4,5}: strides() = {120, 60, 20, 5, 1}
    std::vector<size_t> strides() const;
    std::shared_ptr<base::Buffer> get_buffer() const;
    base::DeviceType device_type() const;

    // 设备管理
    void set_device_type(base::DeviceType device_type) const;

   private:
    size_t size_ = 0;                                     // 张量中数据个数
    std::vector<int32_t> dims_;                           // Tensor的维度
    std::shared_ptr<base::Buffer> buffer_;                // 实质上数据存放位置
    base::DataType data_type_ = base::DataType::Unknown;  // 数据类型
};

template <typename T>
T& Tensor::index(int64_t offset) {
    CHECK_GE(offset, 0) << "Offset must be non-negative";
    CHECK_LT(offset, this->size()) << "Offset out of range";

    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
    CHECK_GE(offset, 0) << "Offset must be non-negative";
    CHECK_LT(offset, this->size()) << "Offset out of range";

    const T& val = *(reinterpret_cast<const T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
T* Tensor::ptr() {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
const T* Tensor::ptr() const {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a "
           "null pointer.";
    return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a "
           "null pointer.";
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor

#endif
