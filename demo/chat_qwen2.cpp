/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-04-02 14:30:00
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-04-02 18:55:08
 * @FilePath: /SiriusxLLM/demo/chat_qwen2.cpp
 * @Description: 基于Qwen2模型实现的ChatML格式聊天功能，提供更多自定义选项
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>
#include "base/base.h"
#include "model/qwen2.h"

// 聊天消息结构体
struct ChatMessage {
    std::string role;    // 角色: system, user, assistant
    std::string content; // 内容
};

// 生成配置结构体
struct GenerationConfig {
    int max_length = 1024;         // 最大生成长度
    int max_context_length = 20480; // 最大上下文长度
};

// 聊天助手类
class ChatAssistant {
public:
    ChatAssistant(const std::string& model_path, 
                  const std::string& tokenizer_path,
                  bool quantized = false,
                  bool use_cuda = true)
        : model_path_(model_path), 
          tokenizer_path_(tokenizer_path),
          quantized_(quantized),
          use_cuda_(use_cuda) {}

    // 初始化模型
    bool init() {
        try {
            model_ = std::make_unique<model::Qwen2Model>(
                base::TokenizerType::EncodeBpe, 
                tokenizer_path_,
                model_path_, 
                quantized_);
                
            auto init_status = model_->init(use_cuda_ ? base::DeviceType::CUDA : base::DeviceType::CPU);
            if (!init_status) {
                LOG(ERROR) << "模型初始化失败: " << init_status.get_err_msg();
                return false;
            }
            return true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "初始化异常: " << e.what();
            return false;
        }
    }

    // 格式化聊天历史为ChatML格式
    std::string format_messages(const std::vector<ChatMessage>& messages) const {
        std::string prompt;
        
        // 按照ChatML格式格式化每条消息
        for (const auto& message : messages) {
            prompt += "<|im_start|>" + message.role + "\n";
            prompt += message.content + "\n";
            prompt += "<|im_end|>\n";
        }
        
        // 添加助手回复的开头部分
        prompt += "<|im_start|>assistant\n";
        
        return prompt;
    }

    // 生成文本回复
    std::string generate(const std::string& prompt, int max_length) {
        auto tokens = model_->encode(prompt);
        int32_t prompt_len = tokens.size();
        LOG_IF(FATAL, tokens.empty()) << "输入tokens为空。";
        
        int32_t pos = 0;
        int32_t next = tokens.at(pos);
        bool is_prompt = true;
        const auto& prompt_embedding = model_->embedding(tokens);
        tensor::Tensor pos_tensor = model_->get_buffer(model::ModelBufferType::InputPos);
        
        std::vector<int32_t> words;
        words.push_back(next);
        
        while (pos < max_length) {
            pos_tensor.index<int32_t>(0) = pos;
            
            if (pos < prompt_len - 1) {
                tensor::Tensor input = model_->fill_input(pos_tensor, prompt_embedding, is_prompt);
                model_->predict(input, pos_tensor, is_prompt, next);
            } else {
                is_prompt = false;
                std::vector<int32_t> current_tokens = {next};
                const auto& token_embedding = model_->embedding(current_tokens);
                tensor::Tensor input = model_->fill_input(pos_tensor, token_embedding, is_prompt);
                model_->predict(input, pos_tensor, is_prompt, next);
            }
            
            if (is_prompt) {
                next = tokens.at(pos + 1);
                words.push_back(next);
            } else {
                words.push_back(next);
                // 检查是否生成了结束标记
                if (pos >= 3) {
                    auto decoded = model_->decode(std::vector<int32_t>(words.end() - 4, words.end()));
                    if (decoded.find("<|im_end|>") != std::string::npos || 
                        decoded.find("<|endoftext|>") != std::string::npos) {
                        break;
                    }
                }
            }
            
            pos += 1;
        }
        
        // 提取生成的回复，去除提示部分
        std::vector<int32_t> response_tokens(words.begin() + prompt_len, words.end());
        std::string response = model_->decode(response_tokens);
        
        // 移除结束标记
        size_t end_pos = response.find("<|im_end|>");
        if (end_pos != std::string::npos) {
            response = response.substr(0, end_pos);
        }
        
        return response;
    }

    // 聊天主方法
    ChatMessage chat(const std::vector<ChatMessage>& messages, const GenerationConfig& config) {
        // 格式化消息
        std::string prompt = format_messages(messages);
        
        // 生成回复
        std::string response_text = generate(prompt, config.max_length);
        
        // 创建并返回回复消息
        ChatMessage response;
        response.role = "assistant";
        response.content = response_text;
        
        return response;
    }

private:
    std::string model_path_;
    std::string tokenizer_path_;
    bool quantized_;
    bool use_cuda_;
    std::unique_ptr<model::Qwen2Model> model_;
};

// 函数声明
void print_help(const std::string& program_name);

// 打印帮助信息
void print_help(const std::string& program_name) {
    std::cout << "用法: " << program_name << " [参数]\n"
              << "必需参数:\n"
              << "  <模型路径>          模型检查点路径\n"
              << "  <分词器路径>        分词器路径\n"
              << "  <是否量化>          是否使用量化模型 (true/false)\n"
              << "  <是否使用CUDA>      是否使用CUDA加速 (true/false)\n"
              << "可选参数:\n"
              << "  [最大生成长度]      生成文本的最大长度 (默认: 1024)\n"
              << "  [系统提示词]        设置AI助手的系统提示词\n"
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
    
    google::InitGoogleLogging(argv[0]);
    
    const char* model_path = argv[1];
    const char* tokenizer_path = argv[2];
    bool quantized = (std::string(argv[3]) == "true");
    bool use_cuda = (std::string(argv[4]) == "true");
    int max_length = (argc > 5) ? std::stoi(argv[5]) : 1024;
    std::string sys_prompt = (argc > 6) ? argv[6] : "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    std::cout << sys_prompt << std::endl;
    // 创建聊天助手实例
    ChatAssistant assistant(model_path, tokenizer_path, quantized, use_cuda);
    
    // 初始化助手
    if (!assistant.init()) {
        LOG(FATAL) << "聊天助手初始化失败!";
        return -1;
    }

    // 初始化聊天历史
    std::vector<ChatMessage> chat_history;
    
    // 添加系统提示
    chat_history.push_back({"system", sys_prompt});
    
    // 设置生成参数
    GenerationConfig gen_config;
    gen_config.max_length = max_length;
    
    std::string user_input;
    bool first_message = true;
    
    std::cout << "初始化完成。输入'quit'结束聊天。\n" << std::endl;
    
    while (true) {
        if (first_message) {
            std::cout << "请输入问题: ";
            first_message = false;
        } else {
            std::cout << "\n请输入问题: ";
        }
        
        // 获取用户输入
        std::getline(std::cin, user_input);
        
        // 检查是否退出
        if (user_input == "quit") {
            break;
        }
        
        // 添加用户消息到历史
        chat_history.push_back({"user", user_input});
        
        // 生成并显示模型回复
        std::cout << "\n助手: " << std::flush;
        
        auto start = std::chrono::steady_clock::now();
        
        // 生成回复
        ChatMessage response = assistant.chat(chat_history, gen_config);
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        std::cout << response.content << std::endl;
        std::cout << "\n[生成时间: " << duration << "秒]" << std::endl;
        
        // 将助手回复添加到历史
        chat_history.push_back(response);
    }
    
    std::cout << "聊天已结束。" << std::endl;
    return 0;
} 