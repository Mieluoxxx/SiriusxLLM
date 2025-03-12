/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-23 18:49:51
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-27 18:13:35
 * @FilePath: /SiriusxLLM/siriusx/include/model/model.h
 * @Description:
 */
#ifndef MODEL_H
#define MODEL_H

#include <op/embedding.h>

#include <memory>
#include <string>

#include "base/base.h"
#include "model/config.h"
#include "model/raw_model_data.h"
#include "op/encode.h"
#include "sampler/sampler.h"
#include "tensor/tensor.h"

// clang-format off
namespace model {
class Model {
   public:
    explicit Model(base::TokenizerType tokenize_type, base::ModelType model_type, std::string token_path,
                   std::string model_path, bool is_quant_model);
    virtual base::Status init(base::DeviceType device_type) = 0;
    virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor, bool is_prompt, int& next) const = 0;
    virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor, int& next) const = 0;

    base::ModelType model_type() const;
    const std::string& token_path() const;
    const std::string& model_path() const;

    virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);
    virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;
    virtual bool is_sentence_ending(int32_t token_idx) const;
    virtual std::string decode(int32_t token_idx) const;
    virtual std::string decode(std::vector<int32_t> token_idxs) const;

    virtual std::vector<int32_t> encode(const std::string& sentence) const;
    virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx, int32_t token_pos) const;
    virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;
    
    virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
        const op::EmbeddingOutput& embedding_output, bool is_prompt) const;

   protected:
    virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);
    virtual base::Status read_model_file();
    virtual base::Status create_encode_layer();
    virtual base::Status gen_model_from_file();
    virtual base::Status generate_model_infos(const ModelConfig& config) const;
    virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

   private:
    virtual void init_mem() = 0;
    virtual base::Status create_layers() = 0;
    virtual void create_param_layers() = 0;
    virtual void create_nonparam_layers() = 0;
    virtual void create_param_quant_layers() = 0;

   protected:
    int32_t group_size_ = 1;
    bool is_quant_model_ = false;
    std::unique_ptr<TransformerConfig> config_;

    std::string token_path_;
    std::string model_path_;
    std::unique_ptr<op::EncodeLayerBase> encode_layer_;
    std::map<ModelBufferType, tensor::Tensor> buffers_;
    std::unique_ptr<sampler::Sampler> sampler_;
    std::shared_ptr<RawModelData> raw_model_data_;
    base::DeviceType device_type_ = base::DeviceType::Unknown;
    base::ModelType model_type_ = base::ModelType::ModelTypeUnknown;
    base::TokenizerType tokenizer_type_ = base::TokenizerType::EncodeUnknown;

};  // model
}  // namespace model

#endif  // MODEL_H