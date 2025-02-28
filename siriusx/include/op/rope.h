/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-17 19:02:51
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 18:28:56
 * @FilePath: /siriusx-infer/siriusx/include/op/rope.h
 * @Description: 审查完成 0228
 */
#ifndef ROPE_H
#define ROPE_H

#include "op/layer.h"

namespace op {
class RoPELayer : public Layer {
   public:
    explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);
    
    base::Status check() const override;
    base::Status forward() override;

   private:
    int32_t dim_ = 0;
    int32_t kv_dim_ = 0;
    int32_t head_size_ = 0;
};
}  // namespace op

#endif  // ROPE_H