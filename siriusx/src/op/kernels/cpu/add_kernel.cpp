/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-26 15:38:07
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-26 15:38:11
 * @FilePath: /SiriusxLLM/siriusx/src/op/kernels/cpu/add_kernel.cpp
 * @Description:
 */
#include "add_kernel.h"

#include <armadillo>

#include "base/base.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& in1, const tensor::Tensor& in2,
                    const tensor::Tensor& out, void* stream) {
    UNUSED(stream);
    CHECK_EQ(in1.is_empty(), false);
    CHECK_EQ(in2.is_empty(), false);
    CHECK_EQ(out.is_empty(), false);

    CHECK_EQ(in1.size(), in2.size());
    CHECK_EQ(in1.size(), out.size());

    arma::fvec in_vec1(const_cast<float*>(in1.ptr<float>()), in1.size(), false, true);
    arma::fvec in_vec2(const_cast<float*>(in2.ptr<float>()), in1.size(), false, true);
    arma::fvec out_vec(const_cast<float*>(out.ptr<float>()), in1.size(), false, true);
    out_vec = in_vec1 + in_vec2;
}
}  // namespace kernel