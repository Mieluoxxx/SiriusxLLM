/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-04-02 16:03:59
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-04-02 18:27:46
 * @FilePath: /SiriusxLLM/demo/demo_qwen2.cpp
 * @Description: 端到端推理性能与资源利用率分析
 */
 #include <base/base.h>
 #include <base/tick.h>
 #include <glog/logging.h>
 #include <chrono>
 #include <numeric>
 #include <vector>
 #include <sys/resource.h>
 
 #include "model/qwen2.h"
 
 // 性能指标结构体
 struct PerformanceMetrics {
     double ttft;              // Time to First Token (ms)
     double tpot;              // Time Per Output Token (ms)
     double throughput;        // 吞吐量 (tokens/s)
     double latency;           // 端到端延迟 (ms)
     double peak_memory;       // 峰值内存占用 (MB)
     int32_t total_tokens;     // 生成的总token数
 };
 
 // 获取当前进程的内存使用情况
 double get_memory_usage() {
     struct rusage usage;
     getrusage(RUSAGE_SELF, &usage);
     return static_cast<double>(usage.ru_maxrss) / 1024.0; // 转换为MB
 }
 
 std::tuple<int32_t, PerformanceMetrics> generate(const model::Qwen2Model& model, 
                                                 const std::string& sentence,
                                                 int total_steps, 
                                                 bool need_output = false) {
     PerformanceMetrics metrics = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
     auto tokens = model.encode(sentence);
     int32_t prompt_len = tokens.size();
     LOG_IF(FATAL, tokens.empty()) << "输入tokens为空。";
 
     int32_t pos = 0;
     int32_t next = tokens.at(pos);
     bool is_prompt = true;
     const auto& prompt_embedding = model.embedding(tokens);
     tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::InputPos);
 
     std::vector<int32_t> words;
     words.push_back(next);
     
     // 性能计时
     auto start_time = std::chrono::steady_clock::now();
     auto first_token_time = start_time;
     bool first_token_generated = false;
     std::vector<double> token_times;
     double peak_memory = 0.0;
 
     while (pos < total_steps) {
         auto token_start_time = std::chrono::steady_clock::now();
         
         pos_tensor.index<int32_t>(0) = pos;
         if (pos < prompt_len - 1) {
             tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
             model.predict(input, pos_tensor, is_prompt, next);
         } else {
             is_prompt = false;
             tokens = std::vector<int32_t>{next};
             const auto& token_embedding = model.embedding(tokens);
             tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
             model.predict(input, pos_tensor, is_prompt, next);
             
             // 记录第一个生成token的时间
             if (!first_token_generated) {
                 first_token_time = std::chrono::steady_clock::now();
                 first_token_generated = true;
             }
         }
 
         // 更新内存使用峰值
         peak_memory = std::max(peak_memory, get_memory_usage());
         
         if (model.is_sentence_ending(next)) {
             break;
         }
         
         if (is_prompt) {
             next = tokens.at(pos + 1);
             words.push_back(next);
         } else {
             words.push_back(next);
             auto token_end_time = std::chrono::steady_clock::now();
             double token_time = std::chrono::duration<double, std::milli>(
                 token_end_time - token_start_time).count();
             token_times.push_back(token_time);
         }
 
         pos += 1;
     }
 
     auto end_time = std::chrono::steady_clock::now();
     
     // 计算性能指标
     metrics.ttft = std::chrono::duration<double, std::milli>(
         first_token_time - start_time).count();
     
     if (!token_times.empty()) {
         metrics.tpot = std::accumulate(token_times.begin(), token_times.end(), 0.0) 
                       / token_times.size();
     }
     
     double total_time = std::chrono::duration<double, std::milli>(
         end_time - start_time).count();
     metrics.latency = total_time;
     metrics.total_tokens = words.size() - prompt_len;
     metrics.throughput = (metrics.total_tokens * 1000.0) / total_time; // tokens/s
     metrics.peak_memory = peak_memory;
 
     if (need_output) {
         printf("%s ", model.decode(words).data());
         fflush(stdout);
     }
     
     return {std::min(pos, total_steps), metrics};
 }
 
 int main(int argc, char* argv[]) {
     if (argc != 3) {
         LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path";
         return -1;
     }
     const char* checkpoint_path = argv[1];
     const char* tokenizer_path = argv[2];
 
     model::Qwen2Model model(base::TokenizerType::EncodeBpe, tokenizer_path,
                             checkpoint_path, true);
     auto init_status = model.init(base::DeviceType::CUDA);
     if (!init_status) {
         LOG(FATAL) << "模型初始化失败: " << init_status.get_err_code();
     }
 
     const std::string& sentence = "你好";
     const int num_runs = 10;  // 进行10次测试取平均值
     std::vector<PerformanceMetrics> all_metrics;
     
     printf("开始性能测试 (运行%d次)...\n", num_runs);
     printf("预热运行...\n");
     generate(model, sentence, 32, false);  // 预热运行
     
     for (int i = 0; i < num_runs; i++) {
         printf("\r运行测试 %d/%d", i+1, num_runs);
         fflush(stdout);
         auto [steps, metrics] = generate(model, sentence, 1024, false);
         all_metrics.push_back(metrics);
     }
     
     // 计算平均指标
     PerformanceMetrics avg_metrics = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
     for (const auto& m : all_metrics) {
         avg_metrics.ttft += m.ttft;
         avg_metrics.tpot += m.tpot;
         avg_metrics.throughput += m.throughput;
         avg_metrics.latency += m.latency;
         avg_metrics.peak_memory = std::max(avg_metrics.peak_memory, m.peak_memory);
         avg_metrics.total_tokens += m.total_tokens;
     }
     
     avg_metrics.ttft /= num_runs;
     avg_metrics.tpot /= num_runs;
     avg_metrics.throughput /= num_runs;
     avg_metrics.latency /= num_runs;
     avg_metrics.total_tokens /= num_runs;
     
     // 输出性能报告
     printf("\n\n========= 性能测试报告 =========\n");
     printf("测试配置:\n");
     printf("- 运行次数: %d\n", num_runs);
     printf("- 输入提示词: \"%s\"\n\n", sentence.c_str());
     
     printf("性能指标 (平均值):\n");
     printf("- TTFT (首token生成时间): %.2f ms\n", avg_metrics.ttft);
     printf("- TPOT (单token生成时间): %.2f ms\n", avg_metrics.tpot);
     printf("- 吞吐量: %.2f tokens/s\n", avg_metrics.throughput);
     printf("- 端到端延迟: %.2f ms\n", avg_metrics.latency);
     printf("- 峰值内存占用: %.2f MB\n", avg_metrics.peak_memory);
     printf("- 平均生成token数: %d\n", avg_metrics.total_tokens);
     printf("==============================\n");
     
     return 0;
 }