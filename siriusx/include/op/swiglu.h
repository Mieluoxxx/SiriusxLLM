#ifndef SWIGLU_H
#define SWIGLU_H

#include "base/base.h"
#include "layer.h"

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