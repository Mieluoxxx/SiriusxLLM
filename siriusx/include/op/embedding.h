/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 19:55:56
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 21:45:47
 * @FilePath: /SiriusxLLM/siriusx/include/op/embedding.h
 * @Description: 
 */
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "op/layer.h"

namespace op {
struct EmbeddingOutput {
    tensor::Tensor input_tokens;
    tensor::Tensor input_embeddings;
    tensor::Tensor input_token_num;
    explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings, tensor::Tensor input_token_nums)
        : input_tokens(std::move(input_tokens)),
          input_embeddings(std::move(input_embeddings)),
          input_token_num(std::move(input_token_nums)) {}
};  // struct EmbeddingOutput

class EmbeddingLayer : public LayerParam {
public:
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size);
    
    base::Status check() const override;
    base::Status forward() override;

   private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};
}  // namespace op

#endif  // EMBEDDING_H