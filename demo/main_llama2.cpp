#include <glog/logging.h>

#include <filesystem>
#include <iostream>

#include "base/base.h"
#include "model/llama2.h"

bool parse_args(int argc, char* argv[], std::string& checkpoint_path,
                std::string& tokenizer_path, bool& is_quantized,
                std::string& prompt, bool& use_cuda);

int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output);

int main(int argc, char* argv[]) {
    google::InitGoogleLogging("SiriusX");
    std::string log_dir = "./log/";

    if (!std::filesystem::exists(log_dir)) {
        std::filesystem::create_directory(log_dir);
    }

    FLAGS_log_dir = log_dir;
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test...\n";

    // 解析命令行参数
    std::string checkpoint_path, tokenizer_path, prompt = "long long ago,";
    bool is_quantized = false;
    bool use_cuda = false;

    if (!parse_args(argc, argv, checkpoint_path, tokenizer_path, is_quantized,
                    prompt, use_cuda)) {
        return -1;
    }

    // 初始化模型
    model::LLama2Model model(base::TokenizerType::EncodeSpe, tokenizer_path,
                             checkpoint_path, is_quantized);
    auto init_status =
        model.init(use_cuda ? base::DeviceType::CUDA : base::DeviceType::CPU);

    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: "
                   << init_status.get_err_msg();
    }

    // 生成文本
    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int steps = generate(model, prompt, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);

    return 0;
}

int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output = false) {
    // 将输入的句子编码为tokens
    auto tokens = model.encode(sentence);
    // 获取prompt的长度
    int32_t prompt_len = tokens.size();
    // 如果tokens为空，则报错
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    // 初始化pos和next
    int32_t pos = 0;
    int32_t next = -1;
    // 初始化is_prompt为true
    bool is_prompt = true;
    // 获取prompt的embedding
    const auto& prompt_embedding = model.embedding(tokens);
    // 获取pos_tensor
    tensor::Tensor pos_tensor =
        model.get_buffer(model::ModelBufferType::InputPos);

    // 初始化words
    std::vector<int32_t> words;
    // 当pos小于total_steps时，循环
    while (pos < total_steps) {
        // 将pos赋值给pos_tensor
        pos_tensor.index<int32_t>(0) = pos;
        // 如果pos小于prompt_len - 1，则填充input
        if (pos < prompt_len - 1) {
            tensor::Tensor input =
                model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            // 进行预测
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
            // 否则，将is_prompt设置为false
            is_prompt = false;
            // 将next设置为tokens的下一个元素
            tokens = std::vector<int32_t>{next};
            // 获取token的embedding
            const auto& token_embedding = model.embedding(tokens);
            // 填充input
            tensor::Tensor input =
                model.fill_input(pos_tensor, token_embedding, is_prompt);
            // 进行预测
            model.predict(input, pos_tensor, is_prompt, next);
        }
        // 如果next是句子的结束，则跳出循环
        if (model.is_sentence_ending(next)) {
            break;
        }
        // 如果is_prompt为true，则将next添加到words中
        if (is_prompt) {
            next = tokens.at(pos + 1);
            words.push_back(next);
        } else {
            // 否则，直接将next添加到words中
            words.push_back(next);
        }
        // pos加1
        pos += 1;
    }
    // 如果need_output为true，则输出结果
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    // 返回pos和total_steps中的最小值
    return std::min(pos, total_steps);
}

// 打印帮助信息
void print_help(const std::string& program_name) {
    std::cout
        << "Usage: " << program_name << " [OPTIONS]\n"
        << "Options:\n"
        << "  --checkpoint_path PATH    Path to the model checkpoint "
           "(required)\n"
        << "  --tokenizer_path PATH     Path to the tokenizer (required)\n"
        << "  --quantized BOOL          Whether the model is quantized "
           "(true/false, default: false)\n"
        << "  --prompt TEXT            Prompt text for generation (default: "
           "'long long ago,')\n"
#ifdef USE_CUDA
        << "  --use_cuda BOOL           Whether to use CUDA (true/false, "
           "default: false)\n"
#endif
        << "  --help                   Show this help message\n";
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
            std::string quantized_str = strip_quotes(argv[++i]);
            is_quantized = (quantized_str == "true");
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = strip_quotes(argv[++i]);
#ifdef USE_CUDA
        } else if (arg == "--use_cuda" && i + 1 < argc) {
            std::string use_cuda_str = strip_quotes(argv[++i]);
            use_cuda = (use_cuda_str == "true");
#endif
        } else if (arg == "--help") {
            print_help(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument or missing value: " << arg << "\n";
            print_help(argv[0]);
            return false;
        }
    }

    // 检查必填参数
    if (checkpoint_path.empty() || tokenizer_path.empty()) {
        std::cerr
            << "Error: --checkpoint_path and --tokenizer_path are required.\n";
        print_help(argv[0]);
        return false;
    }

    return true;
}