#include <base/base.h>

#include "cpu/add_kernel.h"
#include "cpu/matmul_kernel.h"

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
#ifdef USE_CUDA
    else if (device_type == base::DeviceType::CUDA) {
        return matmul_kernel_cuda;
    }
#endif
    else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

}  // namespace kernel