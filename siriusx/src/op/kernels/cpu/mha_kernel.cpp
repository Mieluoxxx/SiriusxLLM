/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-20 20:29:38
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-20 22:31:30
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/mha_kernel.cpp
 * @Description:
 */
#include "mha_kernel.h"

#include <cmath>

#include "../interface.h"
#include "base/alloc.h"
#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
/***
 * @brief 执行多头注意力（Multi-Head Attention）计算的核心函数
 * @param pos 当前解码位置
 * @param head_num 多头注意力头的数量
 * @param layer_index 当前Transformer层索引
 * @param seq_len 序列总长度
 * @param kv_dim key/value的维度（通常等于head_size * num_kv_heads）
 * @param kv_mul key/value头数与query头数的倍数关系（用于GQA/MQA）
 * @param head_size 每个注意力头的维度
 * @param mha_out 输出张量 [hidden_dim]
 * @param query_tensor 查询张量 [head_num, head_size]
 * @param score_tensor 注意力分数缓存 [head_num, seq_len]
 * @param key_cache_tensor 键缓存 [num_layers, seq_len, kv_dim]
 * @param value_cache_tensor 值缓存 [num_layers, seq_len, kv_dim]
 * @param device_type 计算设备类型（CPU/CUDA）
 * @param config CUDA配置（流等），CPU模式下可忽略
 */
void mha_kernel_cpu(int32_t pos, int32_t head_num, int32_t layer_index,
                int32_t seq_len, int32_t kv_dim, int32_t kv_mul,
                int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
    // 计算当前层的缓存偏移量（每层有seq_len个位置，每个位置kv_dim维度）
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    // 缩放因子，用于缩放点积分数（1/sqrt(d_k)）
    float scale = 1.f / std::sqrt(static_cast<float>(head_size));

    // 根据设备类型获取内存分配器
    std::shared_ptr<base::DeviceAllocator> alloc;
    if (device_type == base::DeviceType::CPU) {
        alloc = base::CPUDeviceAllocatorFactory::get_instance();
    }
#ifdef USE_CUDA
    else if (device_type == base::DeviceType::CUDA) {
        alloc = base::CUDADeviceAllocatorFactory::get_instance();
    }
#endif
    else {
        LOG(ERROR) << "MHA ERROR: device type not support";
    }

    // 遍历每个注意力头
    for (int32_t h = 0; h < head_num; ++h) {
        // 当前头的注意力分数缓存指针 [seq_len]
        float* score_head_addr =
            const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
        // 当前头的查询向量指针 [head_size]
        float* query_head_addr =
            const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

        // 将查询向量包装为矩阵视图（行向量）
        tensor::Tensor query_mat(base::DataType::FP32, head_size, false,
                                 nullptr, query_head_addr);
        query_mat.set_device_type(device_type);

        // 计算当前头与所有位置（0到pos）的键的点积
        for (int32_t t = 0; t <= pos; t++) {
            // 计算当前时间步t的键缓存偏移量
            // 结构: [layer][t][group_idx][head_size]（group_idx = h / kv_mul）
            int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
            const float* key_head_addr =
                key_cache_tensor.ptr<float>() + layer_offset + cache_offset;

            // 将键向量包装为列向量矩阵视图
            tensor::Tensor key_mat(base::DataType::FP32, 1, head_size, false,
                                   nullptr, const_cast<float*>(key_head_addr));
            key_mat.set_device_type(device_type);

            // 当前时间步的分数存储位置（score_head_addr[t]）
            tensor::Tensor score_mat(base::DataType::FP32, 1, false, nullptr,
                                     score_head_addr + t);
            score_mat.set_device_type(device_type);

            // 执行查询向量与键向量的点积计算（query_mat * key_mat^T），结果缩放后存入score_mat
            get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
        }

        // 对当前头的注意力分数进行softmax归一化（仅计算到pos位置）
        tensor::Tensor score_head_tensor(base::DataType::FP32, pos + 1, false,
                                         nullptr, score_head_addr);
        score_head_tensor.set_device_type(device_type);
        get_softmax_kernel(device_type)(score_head_tensor,
                                        config ? config->stream : nullptr);

        // 获取当前头的输出指针，并初始化为零
        float* output_head_ptr =
            const_cast<float*>(mha_out.ptr<float>() + h * head_size);
        alloc->memset_zero(output_head_ptr, sizeof(float) * head_size,
                           config ? config->stream : nullptr, false);

        // 将输出向量包装为矩阵视图
        tensor::Tensor output_tensor(base::DataType::FP32, head_size, false,
                                     nullptr, output_head_ptr);
        output_tensor.set_device_type(device_type);

        // 计算当前头对应的值向量缓存偏移量（与键共享分组逻辑）
        int32_t cache_offset = (h / kv_mul) * head_size;
        float* value_head_addr =
            const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset +
            cache_offset;

        // 将值缓存包装为矩阵视图（每个时间步的值向量按行存储）
        tensor::Tensor value_tensor(base::DataType::FP32, head_size, false,
                                    nullptr, value_head_addr);

        // 执行加权求和：output = sum(score_t * value_t) for t in 0..pos
        get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor,
                                          output_tensor, pos, head_size, kv_dim,
                                          config ? config->stream : nullptr);
    }
}

}  // namespace kernel