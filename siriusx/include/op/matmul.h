/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 19:52:54
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-24 15:32:20
 * @FilePath: /siriusx-infer/siriusx/include/op/matmul.h
 * @Description: 
 */
#ifndef MATMUL_H
#define MATMUL_H

#include "base/base.h"
#include "op/layer.h"

namespace op {
class MatmulLayer : public LayerParam {
   public:
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, 
                         bool is_quant_layer = false, bool has_bias = false);
    
    base::Status check() const override;
    base::Status forward() override;
    
    base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr, base::DeviceType device_type);
    tensor::Tensor& get_bias(int32_t idx);
    const tensor::Tensor& get_bias(int32_t idx) const;
    void to_cuda() override;

   private:
    int32_t dim0_;
    int32_t dim1_;
    bool has_bias_ = false;
    std::vector<tensor::Tensor> bias_;
};
}  // namespace op

#endif  // MATMUL_H