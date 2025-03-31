/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-23 21:29:07
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-27 19:55:05
 * @FilePath: /SiriusxLLM/siriusx/include/op/encode.h
 * @Description:
 */
#ifndef ENCODE_H
#define ENCODE_H

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>

#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "nlohmann/json.hpp"
#endif

#include <sentencepiece_processor.h>

#include "op/layer.h"

namespace op {
class EncodeLayerBase : public Layer {
   public:
    explicit EncodeLayerBase(std::string token_model_path, bool has_bos,
                             bool has_eos)
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
    explicit SpeEncodeLayer(std::string token_model_path, bool has_bos,
                            bool has_eos);
    std::vector<int32_t> encode(const std::string& sentence) const override;
    std::string decode(int32_t token_id) const override;
    std::string decode(const std::vector<int32_t>& token_ids) const override;
    bool is_sentence_ending(int32_t token_id) const override;
    int32_t vocab_size() const override;

   private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe_;
};

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT)
class BpeEncodeLayer : public EncodeLayerBase {
   public:
    explicit BpeEncodeLayer(std::string token_model_path, bool has_bos,
                            bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

   protected:
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t stop_token1_ = -1;
    int32_t stop_token2_ = -1;
    int32_t num_token_ = 0;
    std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

class QwenEncodeLayer : public BpeEncodeLayer {
   public:
    explicit QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
};
#endif

}  // namespace op
#endif