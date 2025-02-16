/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-16 21:13:06
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-16 21:15:26
 * @FilePath: /siriusx-infer/siriusx/src/op/kernels/cpu/swiglu_kernel.cpp
 * @Description: 
 */
#include "swiglu_kernel.h"
#include "tensor/tensor.h"
#include <armadillo>


namespace kernel {
    void swiglu_kernel_cpu(const tensor::Tensor& in1, const tensor::Tensor& in2, const tensor::Tensor& out, void* stream) {
        UNUSED(stream);
        CHECK_EQ(in1.is_empty(), false);
        CHECK_EQ(in2.is_empty(), false);
        CHECK_EQ(out.is_empty(), false);

        CHECK(in1.device_type() == base::DeviceType::CPU);
        CHECK(in2.device_type() == base::DeviceType::CPU);
        CHECK(out.device_type() == base::DeviceType::CPU);

        arma::fvec in1_vec(const_cast<float*>(in1.ptr<float>()), in1.size(), false, true);
        arma::fvec in2_vec(const_cast<float*>(in2.ptr<float>()), in2.size(), false, true);
        arma::fvec out_vec(const_cast<float*>(out.ptr<float>()), out.size(), false, true);

        in1_vec %= (1.0f/ (1.0f + arma::exp(-in1_vec)));
        out_vec = in1_vec % in2_vec;
    }
} // namespace kernel