/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-27 20:48:55
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 18:28:09
 * @FilePath: /siriusx-infer/siriusx/include/op/rmsnorm.h
 * @Description: 审查完成 0228
 */
#ifndef RMSNORM_H
#define RMSNORM_H

#include "op/layer.h"

namespace op {
class RMSNormLayer : public LayerParam {
   public:
    explicit RMSNormLayer(base::DeviceType device_type, int32_t dim);
    
    base::Status check() const override;
    base::Status forward() override;

   private:
    int32_t dim_ = 0;
};
}  // namespace op

#endif