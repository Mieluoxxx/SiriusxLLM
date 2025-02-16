#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "layer.h"

namespace op {
struct EmbeddingOutput {
    tensor::Tensor input_tokens;
    tensor::Tensor input_embeddings;
    tensor::Tensor input_token_nums;
    explicit EmbeddingOutput(tensor::Tensor input_tokens,
                             tensor::Tensor input_embeddings,
                             tensor::Tensor input_token_nums)
        : input_tokens(std::move(input_tokens)),
          input_embeddings(std::move(input_embeddings)),
          input_token_nums(std::move(input_token_nums)) {}
};  // struct EmbeddingOutput

class EmbeddingLayer : public LayerParam {
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim,
                            int32_t seq_len, int32_t vocab_size);
    base::Status check() const override;
    base::Status forward() override;

   private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};
}  // namespace op

#endif  // EMBEDDING_H