/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-23 22:15:07
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-27 20:00:32
 * @FilePath: /siriusx-infer/siriusx/src/op/encode.cpp
 * @Description: 
 */
#include "op/encode.h"

#include <sentencepiece_processor.h>

namespace op {
std::string SpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(spe_ != nullptr) << "spe_ is null";
    std::vector<int32_t> token_ids(token_id);
    return this->spe_->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(
    const std::vector<int32_t>& token_ids) const {
    CHECK(spe_ != nullptr) << "spe_ is null";
    return this->spe_->DecodeIds(token_ids);
}

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos,
                               bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using namespace sentencepiece::util;
    spe_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto rc = spe_->Load(token_model_path_);
    if (rc.code() != sentencepiece::util::StatusCode::kOk) {
        LOG(FATAL) << "The token model path is not valid, please check the "
                      "path and type of token model.";
    }
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(spe_ != nullptr) << "spe_ is null";
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

}  // namespace op