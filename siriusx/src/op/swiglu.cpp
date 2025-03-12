/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 21:22:50
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:51:37
 * @FilePath: /SiriusxLLM/siriusx/src/op/swiglu.cpp
 * @Description: 
 */
#include "op/swiglu.h"

#include "base/base.h"
#include "kernels/interface.h"
#include "op/layer.h"

namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::LayerSwiGLU, "SwiGLU"),
      hidden_dim_(hidden_dim) {
    reset_input_size(2);
    reset_output_size(1);
}

base::Status SwiGLULayer::check() const {
    base::Status status;
    const int32_t input_tensor_num = 2;
    for (int32_t i = 0; i < input_tensor_num; i++) {
        status = check_tensor_with_dim(get_input(0), device_type_, data_type_,
                                       hidden_dim_);
        if (!status) {
            LOG(ERROR) << "The input tensor " << std::to_string(i)
                       << " error in the swiglu layer.";
            return status;
        }
    }
    status = check_tensor_with_dim(get_output(0), device_type_, data_type_,
                                   hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the swiglu layer.";
        return status;
    }
    return base::error::Success();
}

base::Status SwiGLULayer::forward() {
    auto status = check();
    if (!status) return status;
    auto in1 = this->get_input(0);
    auto in2 = this->get_input(1);
    auto out = this->get_output(0);
    if (device_type_ == base::DeviceType::CUDA) CHECK(cuda_config_ != nullptr);
    kernel::get_swiglu_kernel(device_type_)(
        in1, in2, out, cuda_config_ ? cuda_config_->stream : nullptr);

    return base::error::Success();
}
}  // namespace op