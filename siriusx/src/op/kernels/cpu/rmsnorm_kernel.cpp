/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-13 17:42:17
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-13 19:53:59
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/rmsnorm_kernel.cpp
 * @Description:
 */
#include "rmsnorm_kernel.h"

#include <armadillo>

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty()) << "input tensor is empty";
    CHECK(!weight.is_empty()) << "weight tensor is empty";
    CHECK(!output.is_empty()) << "output tensor is empty";

    CHECK(input.device_type() == base::DeviceType::CPU &&
          weight.device_type() == base::DeviceType::CPU &&
          output.device_type() == base::DeviceType::CPU)
        << "device type mismatch";

    const float* in_ptr = input.ptr<float>();
    const float* w_ptr = weight.ptr<float>();
    const float* out_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(input.size());

    arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
    arma::fvec w_tensor(const_cast<float*>(w_ptr), dim, false, true);
    arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);

    const float eps = 1e-5f;
    
    // mean: 均方值 + ϵ
    const float mean =
        arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    // rsqrt: 均方值的倒数
    const float rsqrt = 1.f / std::sqrt(mean);
    // %: aramadillo 逐元素相乘
    out_tensor = w_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel