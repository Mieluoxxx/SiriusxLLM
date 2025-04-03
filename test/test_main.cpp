/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-31 03:08:21
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-18 10:28:21
 * @FilePath: /SiriusxLLM/test/test_main.cpp
 * @Description: 
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <filesystem>

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
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
    return RUN_ALL_TESTS();
}

