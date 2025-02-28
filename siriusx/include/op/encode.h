/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-23 21:29:07
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 17:53:34
 * @FilePath: /siriusx-infer/siriusx/include/op/encode.h
 * @Description: 审查完成 0228
 */
#ifndef ENCODE_H
#define ENCODE_H

#include <sentencepiece_processor.h>

#include "op/layer.h"

namespace op {
class EncodeLayerBase : public Layer {
   public:
    explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
        : Layer(base::DeviceType::CPU, LayerType::LayerEncode, "Encode"),
          has_bos_(has_bos),
          has_eos_(has_eos),
          token_model_path_(std::move(token_model_path)) {}

    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;
    virtual std::string decode(int32_t token_id) const = 0;
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;
    virtual bool is_sentence_ending(int32_t token_id) const = 0;
    virtual int32_t vocab_size() const = 0;

   protected:
    bool has_bos_ = true;
    bool has_eos_ = false;
    std::string token_model_path_;
};

class SpeEncodeLayer : public EncodeLayerBase {
   public:
    explicit SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
    std::vector<int32_t> encode(const std::string& sentence) const override;
    std::string decode(int32_t token_id) const override;
    std::string decode(const std::vector<int32_t>& token_ids) const override;
    bool is_sentence_ending(int32_t token_id) const override;
    int32_t vocab_size() const override;

   private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe_;
};

}  // namespace op

#endif