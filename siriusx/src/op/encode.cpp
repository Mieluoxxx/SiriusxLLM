/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-23 22:15:07
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 22:00:07
 * @FilePath: /SiriusxLLM/siriusx/src/op/encode.cpp
 * @Description:
 */
#include "op/encode.h"

#include <glog/logging.h>

#include <fstream>

#include "base/unicode.h"
namespace op {

std::string SpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(spe_ != nullptr);
    std::vector<int32_t> token_ids{token_id};
    return this->spe_->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(
    const std::vector<int32_t>& token_ids) const {
    CHECK(spe_ != nullptr);
    return this->spe_->DecodeIds(token_ids);
}

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos,
                               bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using namespace sentencepiece::util;
    spe_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto rc = spe_->Load(token_model_path_);
    if (rc.code() != StatusCode::kOk) {
        LOG(FATAL) << "The token model path is not valid, please check the "
                      "path and type of token model.";
    }
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(spe_ != nullptr);
    // sentencepiece
    std::vector<int32_t> input_ids = spe_->EncodeAsIds(sentence);
    if (has_bos_) {
        input_ids.insert(input_ids.begin(), spe_->bos_id());
    }
    if (has_eos_) {
        input_ids.push_back(spe_->eos_id());
    }
    return input_ids;
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    CHECK(this->spe_ != nullptr);
    return token_id == this->spe_->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
    CHECK(spe_ != nullptr);
    return spe_->GetPieceSize();
}

#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT)
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos,
                               bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path_);
    CHECK(f.is_open()) << "The token model path is not valid, please check the "
                          "path and type of token model.";
    json data;
    try {
        data = json::parse(f);
    } catch (json::parse_error&) {
        LOG(FATAL) << "The token model path is not valid, please check the "
                      "path and type of token model.";
    }

    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for (const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();
    for (const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    bos_id_ = special_tokens["<|begin_of_text|>"];
    eos_id_ = special_tokens["<|end_of_text|>"];
    stop_token1_ = eos_id_;
    stop_token2_ = special_tokens["<|eot_id|>"];

    num_token_ = encoder.size() + special_tokens.size();
    tiktoken_ =
        std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(this->tiktoken_ != nullptr);
    std::map<std::string, std::string> replacements;
    replacements[" "] = "Ġ";
    std::string s = absl::StrReplaceAll(sentence, replacements);
    auto input_ids = this->tiktoken_->encode(s);

    if (has_bos_) {
        input_ids.insert(input_ids.begin(), bos_id_);
    }
    if (has_eos_) {
        input_ids.push_back(eos_id_);
    }
    return input_ids;
}

std::string BpeEncodeLayer::decode(int32_t token_id) const { return ""; }

std::string BpeEncodeLayer::decode(
    const std::vector<int32_t>& token_ids) const {
    CHECK(this->tiktoken_ != nullptr);
    auto s = tiktoken_->decode(token_ids);
    std::map<std::string, std::string> reverse_replacements;
    reverse_replacements["Ġ"] = " ";
    const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
    return sentence;
}

bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    if (token_id == stop_token1_ || token_id == stop_token2_) {
        return true;
    } else {
        return false;
    }
}

int32_t BpeEncodeLayer::vocab_size() const {
    CHECK(this->tiktoken_ != nullptr);
    return num_token_;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos,
                                 bool has_eos)
    : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path_);

    json data = json::parse(f);
    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for (const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();
    for (const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    bos_id_ = special_tokens["<|im_start|>"];
    eos_id_ = special_tokens["<|im_end|>"];
    stop_token1_ = eos_id_;
    stop_token2_ = special_tokens["<|endoftext|>"];

    num_token_ = encoder.size() + special_tokens.size();
    tiktoken_ =
        std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

#endif
}  // namespace op