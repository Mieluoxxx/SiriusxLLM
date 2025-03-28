/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 19:55:56
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 21:59:32
 * @FilePath: /SiriusxLLM/siriusx/src/op/embedding.cpp
 * @Description: 
 */
#include "op/embedding.h"

#include "base/base.h"
#include "kernels/interface.h"
#include "op/layer.h"

namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim,
                               int32_t seq_len, int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::LayerEmbedding, false, "Embedding") {
    reset_input_size(2);
    reset_output_size(1);
    reset_weight_size(1);
}

base::Status EmbeddingLayer::check() const {
    const auto& input_tensor = get_input(0);
    const auto& token_size = get_input(1).size();
    if (token_size > input_tensor.size()) {
        return base::error::InvalidArgument(
            "The number of input tensor is greater than seq len.");
    }
    base::Status status = check_tensor_with_dim(
        input_tensor, base::DeviceType::CPU, base::DataType::Int32, token_size);

    if (!status) {
        LOG(ERROR) << "The input tensor error in the embedding layer.";
        return status;
    }

    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_,
                                   vocab_size_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_,
                                   token_size, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }
    return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
    base::Status status = check();
    if (!status) return status;
    if (device_type_ == base::DeviceType::CUDA) CHECK(cuda_config_ != nullptr);
    kernel::get_embedding_kernel(device_type_)(
        get_input(0), get_weight(0), get_output(0), vocab_size_,
        cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}

}  // namespace op