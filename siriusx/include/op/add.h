/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-31 03:08:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-07 21:43:42
 * @FilePath: /siriusx-infer/siriusx/include/op/add.h
 * @Description: 
 */
#ifndef OP_ADD_H
#define OP_ADD_H
#include "base/base.h"
#include "layer.h"

namespace op {
class VecAddLayer : public Layer {
   public:
    explicit VecAddLayer(base::DeviceType device_type);
    base::Status check() const override;
    base::Status forward() override;
};
}  // namespace op
#endif  // OP_ADD_H