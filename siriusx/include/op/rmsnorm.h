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