/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 21:44:59
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-05 20:57:12
 * @FilePath: /SiriusxLLM/test/test_op/test_softmax.cpp
 * @Description:
 */
#include <gtest/gtest.h>

#include <armadillo>
#include <chrono>
#include <random>

#include "../src/op/kernels/interface.h"
#include "base/alloc.h"
#include "tensor/tensor.h"

// 原始的armadillo softmax实现
void softmax_3pass(const tensor::Tensor& input, void* stream) {
    int32_t size = static_cast<int32_t>(input.size());
    const float* input_ptr = input.ptr<float>();

    float max_value = *std::max_element(input_ptr, input_ptr + size);

    arma::fvec input_mat(const_cast<float*>(input_ptr), size, false, true);
    input_mat = arma::exp(input_mat - max_value);

    float sum_value = arma::sum(input_mat);
    input_mat = input_mat / sum_value;
}

void softmax_2pass(const tensor::Tensor& input, void* stream) {
    int32_t size = static_cast<int32_t>(input.size());
    const float* input_ptr = input.ptr<float>();
    float* output_ptr = const_cast<float*>(input_ptr);

    // First pass: compute max and sum of exponentials
    float max_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < size; ++i) {
        if (input_ptr[i] > max_value) {
            max_value = input_ptr[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        output_ptr[i] = std::exp(input_ptr[i] - max_value);
        sum_exp += output_ptr[i];
    }

    // Second pass: normalize in-place
    float inv_sum_exp = 1.0f / sum_exp;
    for (int i = 0; i < size; ++i) {
        output_ptr[i] *= inv_sum_exp;
    }
}

// 计算两个张量的最大相对误差
float max_relative_error(const tensor::Tensor& a, const tensor::Tensor& b) {
    int32_t size = static_cast<int32_t>(a.size());
    const float* a_ptr = a.ptr<float>();
    const float* b_ptr = b.ptr<float>();

    float max_error = 0.0f;
    for (int32_t i = 0; i < size; ++i) {
        if (std::abs(a_ptr[i]) > 1e-6f) {
            float rel_error = std::abs((a_ptr[i] - b_ptr[i]) / a_ptr[i]);
            max_error = std::max(max_error, rel_error);
        }
    }
    return max_error;
}

TEST(test_softmax, accuracy) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    const int sizes[] = {10, 100, 1000, 10000};

    for (int size : sizes) {
        // 创建随机输入数据
        tensor::Tensor input(base::DataType::FP32, size, true, alloc_cpu);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (int i = 0; i < size; ++i) {
            input.index<float>(i) = dis(gen);
        }

        // 创建两个输出张量的副本
        tensor::Tensor output1 = input.clone();
        tensor::Tensor output2 = input.clone();

        // 运行两种实现
        softmax_3pass(output1, nullptr);
        softmax_2pass(output2, nullptr);

        // 检查结果是否一致
        float max_error = max_relative_error(output1, output2);
        EXPECT_LT(max_error, 1e-5f) << "Size: " << size;

        LOG(INFO) << "Size: " << size << ", Max relative error: " << max_error;
    }
}

TEST(test_softmax, performance) {
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    const int sizes[] = {100, 1000, 10000, 100000};
    const int iterations = 100;

    for (int size : sizes) {
        // 创建随机输入数据
        tensor::Tensor input(base::DataType::FP32, size, true, alloc_cpu);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (int i = 0; i < size; ++i) {
            input.index<float>(i) = dis(gen);
        }

        // 测量3-pass实现的性能
        auto start1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            tensor::Tensor output1 = input.clone();
            softmax_3pass(output1, nullptr);
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;

        // 测量2-pass实现的性能
        auto start2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            tensor::Tensor output2 = input.clone();
            softmax_2pass(output2, nullptr);
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;

        LOG(INFO) << "Size: " << size;
        LOG(INFO) << "  3-pass: " << elapsed1.count() / iterations << " ms";
        LOG(INFO) << "  2-pass: " << elapsed2.count() / iterations << " ms";
        LOG(INFO) << "  Speedup: " << elapsed1.count() / elapsed2.count()
                  << "x";
    }
}