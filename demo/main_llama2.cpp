#include <glog/logging.h>
#include <filesystem>
#include <iostream>
#include <chrono>

#include "base/base.h"
#include "model/llama2.h"

// 函数声明
bool parse_args(int argc, char* argv[], std::string& checkpoint_path,
                std::string& tokenizer_path, bool& is_quantized,
                std::string& prompt, bool& use_cuda);
int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output);

int main(int argc, char* argv[]) {
    // 初始化日志
    google::InitGoogleLogging("SiriusX");
    std::string log_dir = "./log/";

    if (!std::filesystem::exists(log_dir)) {
        std::filesystem::create_directory(log_dir);
    }

    FLAGS_log_dir = log_dir;
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "开始测试...\n";

    // 解析命令行参数
    std::string checkpoint_path, tokenizer_path;
    std::string prompt = "long long ago,";
    bool is_quantized = false;
    bool use_cuda = false;

    if (!parse_args(argc, argv, checkpoint_path, tokenizer_path, is_quantized,
                    prompt, use_cuda)) {
        return -1;
    }

    // 初始化模型
    model::LLama2Model model(base::TokenizerType::EncodeSpe, 
                             tokenizer_path,
                             checkpoint_path, 
                             is_quantized);
    
    auto device = use_cuda ? base::DeviceType::CUDA : base::DeviceType::CPU;
    auto init_status = model.init(device);

    if (!init_status) {
        LOG(FATAL) << "模型初始化失败，错误信息: " << init_status.get_err_msg();
    }

    // 生成文本
    printf("开始生成文本...\n");
    fflush(stdout);
    
    auto start = std::chrono::steady_clock::now();
    int steps = generate(model, prompt, 128, true);
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\n生成速度: %.2f steps/s\n", static_cast<double>(steps) / duration);
    fflush(stdout);

    return 0;
}

int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output) {
    // 将输入的句子编码为tokens
    auto tokens = model.encode(sentence);
    LOG_IF(FATAL, tokens.empty()) << "编码结果为空";
    
    int32_t prompt_len = tokens.size();
    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;
    std::vector<int32_t> words;
    
    // 获取prompt的embedding和位置张量
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::InputPos);

    // 生成循环
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        
        if (pos < prompt_len - 1) {
            // 处理prompt中的token
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
            // 处理生成的token
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }
        
        // 检查是否生成结束
        if (model.is_sentence_ending(next)) {
            break;
        }
        
        // 添加token到结果中
        if (is_prompt) {
            next = tokens.at(pos + 1);
        }
        words.push_back(next);
        pos += 1;
    }
    
    // 输出生成结果
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    
    return std::min(pos, total_steps);
}

// 打印帮助信息
void print_help(const std::string& program_name) {
    std::cout
        << "用法: " << program_name << " [选项]\n"
        << "选项:\n"
        << "  --checkpoint_path PATH    模型检查点路径 (必需)\n"
        << "  --tokenizer_path PATH     分词器路径 (必需)\n"
        << "  --quantized BOOL          模型是否量化 (true/false, 默认: false)\n"
        << "  --prompt TEXT             生成提示文本 (默认: 'long long ago,')\n"
#ifdef USE_CUDA
        << "  --use_cuda BOOL           是否使用CUDA (true/false, 默认: false)\n"
#endif
        << "  --help                    显示帮助信息\n";
}

// 去掉字符串开头和结尾的双引号
std::string strip_quotes(const std::string& str) {
    if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

// 解析命令行参数
bool parse_args(int argc, char* argv[], std::string& checkpoint_path,
                std::string& tokenizer_path, bool& is_quantized,
                std::string& prompt, bool& use_cuda) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--checkpoint_path" && i + 1 < argc) {
            checkpoint_path = strip_quotes(argv[++i]);
        } else if (arg == "--tokenizer_path" && i + 1 < argc) {
            tokenizer_path = strip_quotes(argv[++i]);
        } else if (arg == "--quantized" && i + 1 < argc) {
            is_quantized = (strip_quotes(argv[++i]) == "true");
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = strip_quotes(argv[++i]);
#ifdef USE_CUDA
        } else if (arg == "--use_cuda" && i + 1 < argc) {
            use_cuda = (strip_quotes(argv[++i]) == "true");
#endif
        } else if (arg == "--help") {
            print_help(argv[0]);
            return false;
        } else {
            std::cerr << "未知参数或缺少值: " << arg << "\n";
            print_help(argv[0]);
            return false;
        }
    }

    // 检查必填参数
    if (checkpoint_path.empty() || tokenizer_path.empty()) {
        std::cerr << "错误: --checkpoint_path 和 --tokenizer_path 为必需参数\n";
        print_help(argv[0]);
        return false;
    }

    return true;
}