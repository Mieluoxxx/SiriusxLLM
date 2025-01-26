#ifndef OP_ADD_H_
#define OP_ADD_H_
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
#endif  // OP_ADD_H_