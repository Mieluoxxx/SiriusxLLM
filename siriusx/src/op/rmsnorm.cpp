/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-13 18:34:55
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-14 22:28:04
 * @FilePath: /siriusx-infer/siriusx/src/op/rmsnorm.cpp
 * @Description:
 */
#include "op/rmsnorm.h"

#include "base/base.h"
#include "kernels/interface.h"

namespace op {
RMSNormLayer::RMSNormLayer(base::DeviceType device_type_, int32_t dim)
    : LayerParam(device_type_, LayerType::LayerRMSNorm, false, "RMSNorm"),
      dim_(dim) {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

base::Status RMSNormLayer::forward() {
    auto status = check();
    if (!status) return status;
    auto input = this->get_input(0);
    auto weight = this->get_weight(0);
    auto output = this->get_output(0);
    if (device_type_ == base::DeviceType::CUDA) CHECK(cuda_config_ != nullptr);
    kernel::get_rmsnorm_kernel(device_type_)(
        input, weight, output, cuda_config_ ? cuda_config_->stream : nullptr);

    return base::error::Success();
}

base::Status RMSNormLayer::check() const {
    auto status =
        check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
        return status;
    }

    status =
        check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
        return status;
    }

    status =
        check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
        return status;
    }
    return base::error::Success();
}
}  // namespace op