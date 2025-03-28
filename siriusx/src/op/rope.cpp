/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-17 21:16:03
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:51:22
 * @FilePath: /SiriusxLLM/siriusx/src/op/rope.cpp
 * @Description: 
 */

#include "op/rope.h"

#include "base/base.h"
#include "kernels/interface.h"
#include "tensor/tensor.h"

namespace op {
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size)
    : Layer(device_type, LayerType::LayerRoPE, "RoPE"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) {
    reset_input_size(5);
    reset_output_size(1);
}

base::Status RoPELayer::check() const {
    // pos tensor
    auto status = check_tensor_with_dim(get_input(2), base::DeviceType::CPU,
                                        base::DataType::Int32, 1);
    if (!status) {
        LOG(ERROR) << "The input tensor 2 error in the add layer.";
        return status;
    }

    status =
        check_tensor_with_dim(get_input(1), device_type_, data_type_, kv_dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor 1 error in the add layer.";
        return status;
    }

    status =
        check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor 0 error in the add layer.";
        return status;
    }
    return base::error::Success();
}

base::Status RoPELayer::forward() {
    base::Status status = check();
    if (!status) return status;

    tensor::Tensor input_q = this->get_input(0);
    tensor::Tensor input_k = this->get_input(1);
    tensor::Tensor input_pos = this->get_input(2);

    tensor::Tensor sin_cache = this->get_input(3);
    tensor::Tensor cos_cache = this->get_input(4);

    if (device_type_ == base::DeviceType::CUDA) CHECK(cuda_config_ != nullptr);

    kernel::get_rope_kernel(device_type_)(
        dim_, kv_dim_, head_size_, input_q, input_k, input_pos, sin_cache,
        cos_cache, cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}

}  // namespace op
