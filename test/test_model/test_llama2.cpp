/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-03-02 13:39:26
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-02 13:51:56
 * @FilePath: /SiriusxLLM/test/test_model/test_llama2.cpp
 * @Description: 
 */
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/base.h"
#include "model/llama2.h"
#include "tensor/tensor.h"

TEST(test_llama_model, cpu1) {
  using namespace base;
  std::shared_ptr<base::CPUDeviceAllocator> alloc = std::make_shared<base::CPUDeviceAllocator>();

  const char* checkpoint_path = "/home/moguw/workspace/SiriusxLLM/tmp/stories42M.bin";
  const char* tokenizer_path = "/home/moguw/workspace/SiriusxLLM/tmp/tokenizer.model";
  model::LLama2Model model(base::TokenizerType::EncodeSpe, tokenizer_path, checkpoint_path, false);
  auto status = model.init(base::DeviceType::CPU);

  if (status) {
    std::string sentence = "Hi";  // prompts
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::InputPos);
    bool is_prompt = true;
    auto tokens = model.encode(sentence);
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
    int32_t next = -1;
    const auto s = model.forward(input, pos_tensor, next);
    const float* logits = model.get_buffer(model::ModelBufferType::ForwardOutput).ptr<float>();
    for (int i = 0; i < 10; ++i) {
      LOG(INFO) << "logits[" << i << "] = " << logits[i];
    }
  }
}