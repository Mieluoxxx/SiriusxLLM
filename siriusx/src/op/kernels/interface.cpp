/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-31 03:08:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-17 21:14:32
 * @FilePath: /siriusx-infer/siriusx/src/op/kernels/interface.cpp
 * @Description:
 */
#include <base/base.h>

#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"

#ifdef USE_CUDA
#include "cuda/add_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
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
    #ifdef USE_CUDA
    else if (device_type == base::DeviceType::CUDA) {
        return rmsnorm_kernel_cuda;
    }
    #endif
    else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return embedding_kernel_cpu;
    }
    // #ifdef USE_CUDA
    //     else if (device_type == base::DeviceType::CUDA) {
    //         return embedding_kernel_cuda;
    //     }
    // #endif
    else {
        LOG(FATAL) << "Unknown device type for get a embedding kernel.";
        return nullptr;
    }
}

SwiGLUKernel get_swiglu_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return swiglu_kernel_cpu;
    }
    // #ifdef USE_CUDA
    //     else if (device_type == base::DeviceType::CUDA) {
    //         return swiglu_kernel_cuda;
    //     }
    // #endif
    else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return softmax_inplace_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get an softmax kernel.";
        return nullptr;
    }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return rope_kernel_cpu;
    }
    // #ifdef USE_CUDA
    //     else if (device_type == base::DeviceType::CUDA) {
    //         return rope_kernel_cuda;
    //     }
    // #endif
    else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return scale_inplace_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get an scale kernel.";
        return nullptr;
    }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return scale_sum_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get an scale sum kernel.";
        return nullptr;
    }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return mha_kernel_cpu;
    }
// #ifdef USE_CUDA
//     else if (device_type == base::DeviceType::CUDA) {
//         return mha_kernel_cuda
//     }
// #endif
    else {
        LOG(FATAL) << "Unknown device type for get an mha kernel.";
        return nullptr;
    }
}

}  // namespace kernel