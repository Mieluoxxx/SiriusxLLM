// #include "base/tick.h"
#include <glog/logging.h>

#include <filesystem>

#include "base/base.h"
#include "model/llama2.h"

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
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
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

int main(int argc, char* argv[]) {
    google::InitGoogleLogging("SiriusX");
    // 日志目录路径
    std::string log_dir = "./log/";

    // 检查日志目录是否存在，如果不存在则创建
    if (!std::filesystem::exists(log_dir)) {
        std::filesystem::create_directory(log_dir);
    }

    // 设置日志目录
    FLAGS_log_dir = log_dir;
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test...\n";

    printf("number of argc: %d\n", argc);
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];

    model::LLama2Model model(base::TokenizerType::EncodeSpe, tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::CPU);
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_msg();
    }
    const std::string& sentence = "long long ago,";

    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int steps = generate(model, sentence, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
