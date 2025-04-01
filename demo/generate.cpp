/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-03-16 13:29:53
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-04-01 08:06:37
 * @FilePath: /SiriusxLLM/demo/generate.cpp
 * @Description: 
 */
#include <base/base.h>
#include <glog/logging.h>

#include "model/llama2.h"

// 生成函数，用于生成指定长度的文本
int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output = false) {
    // 对输入的句子进行编码
    auto tokens = model.encode(sentence);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor =
        model.get_buffer(model::ModelBufferType::InputPos);

    std::vector<int32_t> words;
    // 循环生成文本，直到达到指定步数或遇到句子结束符
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        if (pos < prompt_len - 1) {
            // 如果还在生成提示词，则使用提示词的嵌入
            tensor::Tensor input =
                model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
            // 如果已经生成完提示词，则开始生成新的词
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input =
                model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }
        // 如果生成的词是句子结束符，则结束生成
        if (model.is_sentence_ending(next)) {
            break;
        }
        // 将生成的词添加到结果中
        if (is_prompt) {
            next = tokens.at(pos + 1);
            words.push_back(next);
        } else {
            words.push_back(next);
        }

        pos += 1;
    }
    // 如果需要输出结果，则进行解码并输出
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    // 返回生成的步数
    return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];

    // 初始化模型
    model::LLama2Model model(base::TokenizerType::EncodeSpe, tokenizer_path,
                             checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::CUDA);
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: "
                   << init_status.get_err_msg();
    }
    const std::string& sentence = "a";

    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    // 生成指定长度的文本
    int steps = generate(model, sentence, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
