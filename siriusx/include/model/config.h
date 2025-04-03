/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-23 18:41:52
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 21:44:43
 * @FilePath: /SiriusxLLM/siriusx/include/model/config.h
 * @Description: 模型配置相关的头文件，定义了模型的基本参数结构
 */
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <ostream>

namespace model {
/**
 * @brief 模型基础配置结构体
 * 包含模型的基本参数设置
 */
struct ModelConfig {
    int32_t dim = 0;        // 模型的嵌入维度
    int32_t hidden_dim = 0; // 前馈网络隐藏层维度
    int32_t layer_num = 0;  // 模型层数
    int32_t head_num = 0;   // 注意力头数量
    int32_t kv_head_num = 0;// KV注意力头数量（用于分组注意力机制）
    int32_t vocab_size = 0; // 词表大小
    int32_t seq_len = 0;    // 序列最大长度
};

/**
 * @brief Transformer模型详细配置结构体
 * 包含更多Transformer架构相关的参数
 */
struct TransformerConfig {
    int32_t kv_dim_ = 0;           // KV投影维度
    int32_t kv_mul_ = 0;           // KV维度倍数
    int32_t head_size_ = 0;        // 每个注意力头的维度大小
    int32_t vocab_size_ = 0;       // 词表大小

    int32_t dim_ = 0;              // 模型的嵌入维度
    int32_t hidden_dim_ = 0;       // 前馈网络隐藏层维度
    int32_t layer_num_ = 0;        // 模型层数
    int32_t head_num_ = 0;         // 注意力头数量
    int32_t kv_head_num_ = 0;      // KV注意力头数量
    int32_t seq_len_ = 0;          // 序列最大长度
    bool is_shared_weight_ = false;// 是否共享输入输出权重

    /**
     * @brief 输出流运算符重载
     * 用于打印TransformerConfig的内容
     */
    friend std::ostream& operator<<(std::ostream& os, const TransformerConfig& obj) {
        return os << "\nkv_dim: " << obj.kv_dim_ << "\nkv_mul_: " << obj.kv_mul_
                  << "\n"
                  << "head_size: " << obj.head_size_
                  << "\nvocab_size_: " << obj.vocab_size_ << "\n"
                  << "dim: " << obj.dim_ << "\nhidden_dim_: " << obj.hidden_dim_
                  << "\n"
                  << "layer_num: " << obj.layer_num_
                  << "\nhead_num_: " << obj.head_num_ << "\n"
                  << "kv_head_num: " << obj.kv_head_num_
                  << "\nseq_len_: " << obj.seq_len_ << "\n"
                  << "is_shared_weight: " << obj.is_shared_weight_;
    }
};
}  // namespace model

#endif  // MODEL_CONFIG_H