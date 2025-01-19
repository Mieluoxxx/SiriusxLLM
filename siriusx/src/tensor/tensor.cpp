/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-15 21:36:02
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-17 20:14:22
 * @FilePath: /SiriusX-infer/siriusx/src/tensor/tensor.cpp
 * @Description:
 */
#include "tensor/tensor.h"

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <numeric>

#include "base/alloc.h"
#include "base/base.h"

namespace tensor {
/***
 * @description: 计算多维张量总元素数量
 * @param { T} 迭代器类型
 * @param {Tp} 初始值的类型
 * @return {*} size_t 总元素数量
 */
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
    if (begin >= end) return 0;
    // std::accumulate: 作用是对容器中的元素进行累乘，求得张量总元素数量
    // 张量形状为 (2, 3, 4) => 1 * 2 * 3 * 4 = 24
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());

    return size;
}

// 计算数据类型大小
static size_t data_type_size(base::DataType data_type) {
    switch (data_type) {
        case base::DataType::FP32:
            return 4;
        case base::DataType::Int8:
            return 1;
        case base::DataType::Int32:
            return 4;
        default: {
            LOG(FATAL) << "Unknown data type size for " << int(data_type);
            return 0;
        }
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    size_ = dim0;
    if (need_alloc && alloc) {  // 需要分配内存
        allocate(alloc);
    } else {  // 不需要分配内存, 使用传入的指针
        if (ptr != nullptr) {
            CHECK(need_alloc == false) << "The need_alloc is is true when ptr "
                                          "parameter is not a null pointer.";
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc,
               void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
               int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
               int32_t dim2, int32_t dim3, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc,
               void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
    size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

void Tensor::to_cpu() {
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type = buffer_->device_type();

    if (device_type == base::DeviceType::Unknown) {
        LOG(ERROR) << "The device type of the tensor is unknown.";
    }
    // TODO to_cpu时CUDA2CPU
    else if (device_type == base::DeviceType::CUDA) {
        LOG(ERROR) << "The device type of the tensor is CUDA, which is not ";
    } else {
        LOG(INFO) << "The device type of the tensor is already cpu.";
    }
}

size_t Tensor::size() const { return this->size_; }

int32_t Tensor::get_dim(int32_t index) const {
    CHECK_GE(index, 0) << "The index must be greater than or equal to 0.";
    CHECK_LT(index, this->dims_.size());

    return this->dims_.at(size_);  // .at()会进行越界检查
}

base::DeviceType Tensor::device_type() const {
    if (!buffer_) {
        return base::DeviceType::Unknown;
    }

    return buffer_->device_type();
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
    if (!buffer) {
        LOG(ERROR)
            << "The buffer parameter in the assign function is null pointer!";
        return false;
    }
    if (buffer_) {
        if (buffer_->device_type() != buffer->device_type()) {
            LOG(ERROR) << "The device type of the new buffer is different from "
                          "the original one.";
        }
    }

    size_t byte_size = buffer->byte_size();
    if (byte_size > buffer->byte_size()) {
        LOG(ERROR) << "The size of buffer is too small for the tensor!";
        return false;
    }
    buffer_ = buffer;

    return true;
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                      bool need_realloc) {
    if (!allocator) {
        LOG(ERROR) << "The allocator parameter in the allocate function is "
                      "null pointer!";
        return false;
    }

    size_t byte_size = this->byte_size();
    if (!byte_size) {
        LOG(ERROR) << "The byte_size parameter in the allocate function is "
                      "equal to zero!";
        return false;
    }

    if (buffer_ && byte_size <= buffer_->byte_size()) {
        if (!need_realloc) {
            return true;
        }
    }

    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
    if (!buffer_->ptr()) {
        LOG(ERROR) << "Failed to allocate memory for the tensor!";
        return false;
    }

    return true;
}

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

// const: 将 Tensor 的设备类型同步给 buffer 类型
void Tensor::set_device_type(base::DeviceType device_type) const {
    if (buffer_) {
        buffer_->set_device_type(device_type);
    }
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

base::DataType Tensor::data_type() const { return data_type_; }

void Tensor::reshape(const std::vector<int32_t>& dims) {
    size_t size = reduce_dimension(dims_.begin(), dims_.end(), 1);
    if (!buffer_) {
        this->dims_ = dims;
        this->size_ = size;
        return;
    }
    if (size > size_) {
        auto new_buffer = std::make_shared<base::Buffer>(
            size * base::DataTypeSize(this->data_type_), buffer_->allocator());
    }
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

Tensor Tensor::clone() const {
    Tensor new_tensor = *this;
    size_t byte_size = this->byte_size();

    auto allocator = buffer_->allocator();
    new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    new_tensor.buffer_->copy_from(buffer_.get());

    return new_tensor;
}

size_t Tensor::byte_size() const {
    return this->size() * base::DataTypeSize(data_type_);
}

// strides: 每个维度上的步长
std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> strides;
    if (!dims_.empty()) {
        for (int32_t i = 0; i < dims_.size() - 1; i++) {
            size_t stride =
                reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
            strides.push_back(stride);
        }
        strides.push_back(1);
    }
    return strides;
}

bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                         base::DataType data_type, bool need_alloc, void* ptr) {
    if (!alloc && !need_alloc) {
        std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
            data_type_size(data_type) * size_, nullptr, ptr, true);
        this->buffer_ = buffer;
    } else {
        allocate(alloc, true);
    }
}
}  // namespace tensor