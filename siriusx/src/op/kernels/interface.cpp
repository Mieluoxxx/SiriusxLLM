/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-31 03:08:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-13 19:53:04
 * @FilePath: /siriusx-infer/siriusx/src/op/kernels/interface.cpp
 * @Description: 
 */
#include <base/base.h>

#include "cpu/add_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/rmsnorm_kernel.h"

#ifdef USE_CUDA
#include "cuda/add_kernel.cuh"
#endif

#include "interface.h"

namespace kernel {

AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return add_kernel_cpu;
    }
#ifdef USE_CUDA
    else if (device_type == base::DeviceType::CUDA) {
        return add_kernel_cuda;
    }
#endif
    else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return matmul_kernel_cpu;
    }
// #ifdef USE_CUDA
//     else if (device_type == base::DeviceType::CUDA) {
//         return matmul_kernel_cuda;
//     }
// #endif
    else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return rmsnorm_kernel_cpu;
    }
// #ifdef USE_CUDA
//     else if (device_type == base::DeviceType::CUDA) {
//         return rmsnorm_kernel_cuda;
//     }
// #endif
    else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

}  // namespace kernel