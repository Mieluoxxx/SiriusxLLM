/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-31 03:08:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 18:58:06
 * @FilePath: /siriusx-infer/siriusx/src/op/add.cpp
 * @Description: 审查完成 0228
 */
#include "op/add.h"

#include "base/base.h"
#include "kernels/interface.h"

namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::LayerAdd, "Add") {
    reset_input_size(2);
    reset_output_size(1);
}

base::Status VecAddLayer::check() const {
    tensor::Tensor in1 = this->get_input(0);
    tensor::Tensor in2 = this->get_input(1);
    int32_t size = in1.size();
    base::Status status;
    status = check_tensor_with_dim(in1, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The input tensor 1 error in the add layer.";
        return status;
    }

    status = check_tensor_with_dim(in2, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The input tensor 2 error in the add layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the add layer.";
        return status;
    }
    return base::error::Success();
}

base::Status VecAddLayer::forward() {
    auto status = this->check();
    if (!status) {
        return status;
    }
    auto in1 = this->get_input(0);
    auto in2 = this->get_input(1);
    auto out = this->get_output(0);
    if (device_type_ == base::DeviceType::CUDA) {
        CHECK(cuda_config_ != nullptr);
    }
    kernel::get_add_kernel(device_type_)(in1, in2, out, cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}
}  // namespace op