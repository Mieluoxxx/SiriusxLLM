/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 21:10:11
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-24 15:33:15
 * @FilePath: /SiriusxLLM/siriusx/include/op/swiglu.h
 * @Description: 
 */
#ifndef SWIGLU_H
#define SWIGLU_H

#include "base/base.h"
#include "op/layer.h"

namespace op {
class SwiGLULayer : public Layer {
   public:
    explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

    base::Status check() const override;
    base::Status forward() override;

   private:
    int32_t hidden_dim_ = 0;
};
}  // namespace op

#endif  // SWIGLU_H