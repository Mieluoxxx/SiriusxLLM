/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-17 19:11:51
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-07 14:41:45
 * @FilePath: /siriusx-infer/siriusx/src/op/kernels/cpu/rope_kernel.cpp
 * @Description: 
 */
#include "rope_kernel.h"

#include <cmath>

#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {

// 计算sin和cos缓存，用于后续的旋转位置编码
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache,
                            float* cos_cache) {
    // 遍历每个序列位置
    for (int pos = 0; pos < max_seq_len; pos++) {
        // 遍历每个头维度
        for (int head_dim = 0; head_dim < head_size; head_dim++) {
            // 计算当前头维度的频率
            // 公式: freq = 1 / 10000^(head_dim / head_size)
            float freq =
                1.0f / std::pow(10000.0f, static_cast<float>(head_dim) /
                                              static_cast<float>(head_size));

            // 计算当前位置对应的角度值
            // 公式: angle = pos * freq
            float val = static_cast<float>(pos) * freq;

            // 计算当前角度的余弦和正弦值
            // 公式: cos(angle) 和 sin(angle)
            float fcr = cosf(val);  // 余弦值
            float fci = sinf(val);  // 正弦值

            // 将计算得到的余弦和正弦值存入缓存中
            // 缓存索引: pos * head_size + head_dim
            *(sin_cache + pos * head_size + head_dim) = fci;
            *(cos_cache + pos * head_size + head_dim) = fcr;
        }
    }
}

// 执行旋转位置编码的核心核函数
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache,
                     const tensor::Tensor& cos_cache, void* stream) {
    // 忽略stream参数
    UNUSED(stream);

    // 获取当前的位置信息
    // 假设input_pos是一个包含单个整数的张量，表示当前序列位置
    const int32_t pos = *input_pos.ptr<int32_t>(0);

    // 遍历每个维度
    for (int32_t i = 0; i < dim; i++) {
        // 计算当前维度对应的头维度索引
        // 公式: head_dim = i % head_size
        int32_t head_dim = i % head_size;

        // 从缓存中获取当前位置和头维度对应的余弦和正弦值
        // 缓存索引: pos * head_size + head_dim
        float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim);
        float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim);

        // 计算当前维度需要旋转的次数
        // 如果当前维度小于kv_dim，则旋转两次；否则旋转一次
        int32_t rotn = (i < kv_dim) ? 2 : 1;

        // 遍历每次旋转
        for (int32_t v = 0; v < rotn; v++) {
            // 根据旋转类型选择输入张量（query或key）
            // 如果是第一次旋转（v == 0），则操作query；否则操作key
            float* vec = const_cast<float*>((v == 0) ? input_q.ptr<float>()
                                                     : input_k.ptr<float>());

            // 获取当前维度的两个连续值，通常用于表示一个向量
            float v0 = vec[i];      // 第一个值
            float v1 = vec[i + 1];  // 第二个值

            // 执行旋转操作
            // 公式:
            // new_v0 = v0 * cos(angle) - v1 * sin(angle)
            // new_v1 = v0 * sin(angle) + v1 * cos(angle)
            vec[i] = v0 * fcr - v1 * fci;      // 新的v0（cos部分）
            vec[i + 1] = v0 * fci + v1 * fcr;  // 新的v1（sin部分）
        }
    }
}

}  // namespace kernel