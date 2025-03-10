/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-20 20:21:01
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-27 10:17:48
 * @FilePath: /SiriusxLLM/siriusx/include/op/mha.h
 * @Description: 
 */
#ifndef MHA_H
#define MHA_H

#include <base/cuda_config.h>

#include "base/base.h"
#include "op/layer.h"

namespace op {
class MultiHeadAttention : public Layer {
   public:
   /**
    * @brief 构造一个多头注意力机制（MultiHeadAttention）对象。
    *
    * @param device_type 设备类型，指定计算将在哪种设备上执行（如CPU、GPU等）。
    * @param layer_index 层索引，标识当前注意力层在网络中的位置。
    * @param kv_mul kv与q的比值，作用效果参考GQA
    * @param kv_dim Key和Value的维度大小。
    * @param seq_len 序列长度，表示输入序列的时间步长或token数量。
    * @param head_num 注意力头的数量，决定模型并行处理注意力机制的能力。
    * @param head_size 每个注意力头的大小，影响模型的表示能力。
    */
    explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index, int32_t kv_mul,
                                int32_t kv_dim, int32_t seq_len, int32_t head_num, int32_t head_size);
    
    // 检查MultiHeadAttention类的状态
    base::Status check() const override;
    // 前向传播
    base::Status forward() override;

    // 设置位置
    void set_pos(int32_t pos);
    // 设置层索引
    void set_layer_idx(int32_t layer_idx);

   private:
    // 层索引
    int32_t layer_index_ = 0;
    // 位置
    int32_t pos_ = 0;
    // kv乘数
    int32_t kv_mul_ = 0;
    // kv维度
    int32_t kv_dim_ = 0;
    // 序列长度
    int32_t seq_len_ = 0;
    // 头数
    int32_t head_num_ = 0;
    // 头大小
    int32_t head_size_ = 0;
};
}  // namespace op

#endif  // MHA_H