/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-03-16 13:29:53
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-18 14:30:03
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/rmsnorm_kernel.cpp
 * @Description: 
 */
#include "rmsnorm_kernel.h"

#include <armadillo>

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input,
                        const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::CPU &&
          weight.device_type() == base::DeviceType::CPU &&
          output.device_type() == base::DeviceType::CPU);

    const float* in_ptr = input.ptr<float>();
    const float* wei_ptr = weight.ptr<float>();
    const float* out_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(input.size());

    arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
    arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
    arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

#ifdef QWEN2_SUPPORT
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif

    const float mean =
        arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    const float rsqrt = 1.f / std::sqrt(mean);
    out_tensor = wei_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel