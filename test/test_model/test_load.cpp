/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-03-02 15:59:34
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-02 16:47:09
 * @FilePath: /SiriusxLLM/test/test_model/test_load.cpp
 * @Description:
 */
#include <fcntl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include "model/config.h"

TEST(test_load, load_model_config) {
    std::string model_path = "/home/moguw/workspace/SiriusxLLM/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY);
    ASSERT_NE(fd, -1);

    FILE* file = fopen(model_path.data(), "rb");
    ASSERT_NE(file, nullptr);

    auto config = model::ModelConfig();
    fread(&config, sizeof(model::ModelConfig), 1, file);
    ASSERT_EQ(config.dim, 16);
    ASSERT_EQ(config.hidden_dim, 128);
    ASSERT_EQ(config.layer_num, 256);
}

TEST(test_load, load_model_weight) {
    std::string model_path = "/home/moguw/workspace/SiriusxLLM/tmp/test.bin";
    int32_t fd = open(model_path.data(), O_RDONLY);
    ASSERT_NE(fd, -1);

    FILE* file = fopen(model_path.data(), "rb");
    ASSERT_NE(file, nullptr);

    auto config = model::ModelConfig{};
    fread(&config, sizeof(model::ModelConfig), 1, file);

    fseek(file, 0, SEEK_END);
    auto file_size = ftell(file);

    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    float* weight_data = reinterpret_cast<float*>(static_cast<int8_t*>(data) + sizeof(model::ModelConfig));

    for (int i = 0; i < config.dim * config.hidden_dim; ++i) {
        ASSERT_EQ(*(weight_data + i), float(i));
    }
}