/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-24 20:42:42
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-03-05 19:40:22
 * @FilePath: /SiriusxLLM/siriusx/src/sampler/argmax_sampler.cpp
 * @Description: 
 */
#include "sampler/argmax_sampler.h"

#include <glog/logging.h>

#include <algorithm>

#include "base/base.h"

#ifdef USE_CUDA
#include "../op/kernels/cuda/argmax_kernel.cuh"
#endif

namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
    if (device_type_ == base::DeviceType::CPU) {
        size_t next =
            std::distance(logits, std::max_element(logits, logits + size));
        return next;
    }
#if USE_CUDA
    else if (device_type_ == base::DeviceType::CUDA) {
        size_t next = kernel::argmax_kernel_cuda(logits, size, stream);
        return next;
    }
#endif
    else {
        LOG(FATAL) << "Unsupported device type for ArgmaxSampler";
    }
}
}  // namespace sampler