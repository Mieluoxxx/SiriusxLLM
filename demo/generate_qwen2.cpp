/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-04-02 14:30:00
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-04-02 18:55:08
 * @FilePath: /SiriusxLLM/demo/gen_qwen2.cpp
 * @Description: 基于Qwen2模型实现的文本生成功能
 */

#include <glog/logging.h>

#include <chrono>
#include <filesystem>
#include <iostream>

#include "base/base.h"
#include "model/qwen2.h"

// 函数声明
void print_help(const std::string& program_name);
int32_t generate(const model::Qwen2Model& model, const std::string& sentence,
                 int total_steps, bool need_output);

// 打印帮助信息
void print_help(const std::string& program_name) {
    std::cout << "用法: " << program_name << " [参数]\n"
              << "必需参数:\n"
              << "  <模型路径>          模型检查点路径\n"
              << "  <分词器路径>        分词器路径\n"
              << "  <是否量化>          是否使用量化模型 (true/false)\n"
              << "  <是否使用CUDA>      是否使用CUDA加速 (true/false)\n"
              << "可选参数:\n"
              << "  [最大生成长度]      生成文本的最大长度 (默认: 128)\n"
              << "  [提示词]            生成文本的起始提示词 (默认: '你好，请给我讲个故事')\n"
              << "  -h                  显示帮助信息\n";
}

int main(int argc, char* argv[]) {
    // 检查是否请求帮助
    if (argc == 2 && std::string(argv[1]) == "-h") {
        print_help(argv[0]);
        return 0;
    }

    if (argc < 5) {
        print_help(argv[0]);
        return -1;
    }
    
    // 初始化日志
    google::InitGoogleLogging("SiriusX");
    std::string log_dir = "./log/";

    if (!std::filesystem::exists(log_dir)) {
        std::filesystem::create_directory(log_dir);
    }

    FLAGS_log_dir = log_dir;
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "开始测试...\n";

    // 解析参数
    const char* model_path = argv[1];
    const char* tokenizer_path = argv[2];
    bool is_quantized = (std::string(argv[3]) == "true");
    bool use_cuda = (std::string(argv[4]) == "true");
    int max_length = (argc > 5) ? std::stoi(argv[5]) : 128;
    std::string prompt = (argc > 6) ? argv[6] : "你好，请给我讲个故事";

    // 初始化模型
    model::Qwen2Model model(base::TokenizerType::EncodeBpe, tokenizer_path,
                             model_path, is_quantized);

    auto device = use_cuda ? base::DeviceType::CUDA : base::DeviceType::CPU;
    auto init_status = model.init(device);

    if (!init_status) {
        LOG(FATAL) << "模型初始化失败，错误信息: " << init_status.get_err_msg();
    }

    // 生成文本
    printf("开始生成文本...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    int steps = generate(model, prompt, max_length, true);
    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\n生成速度: %.2f steps/s\n", static_cast<double>(steps) / duration);
    fflush(stdout);

    return 0;
}

int32_t generate(const model::Qwen2Model& model, const std::string& sentence,
                 int total_steps, bool need_output) {
    // 将输入的句子编码为tokens
    auto tokens = model.encode(sentence);
    LOG_IF(FATAL, tokens.empty()) << "编码结果为空";

    int32_t prompt_len = tokens.size();
    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    bool is_prompt = true;
    std::vector<int32_t> words;
    words.push_back(next);

    // 获取prompt的embedding和位置张量
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor =
        model.get_buffer(model::ModelBufferType::InputPos);

    // 生成循环
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;

        if (pos < prompt_len - 1) {
            // 处理prompt中的token
            tensor::Tensor input =
                model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
            // 处理生成的token
            is_prompt = false;
            std::vector<int32_t> current_tokens = {next};
            const auto& token_embedding = model.embedding(current_tokens);
            tensor::Tensor input =
                model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }

        // 处理token添加
        if (is_prompt) {
            next = tokens.at(pos + 1);
            words.push_back(next);
        } else {
            words.push_back(next);
        }

        // 检查是否生成结束
        if (model.is_sentence_ending(next)) {
            break;
        }

        pos += 1;
    }

    // 输出生成结果
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }

    return std::min(pos, total_steps);
} 