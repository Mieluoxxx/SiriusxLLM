#include "sampler/argmax_sampler.h"

#include <glog/logging.h>

#include <algorithm>

#include "base/base.h"
namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
    if (device_type_ == base::DeviceType::CPU) {
        size_t next =
            std::distance(logits, std::max_element(logits, logits + size));
        return next;
    }
#if USE_CUDA
    else if {
        size_t next = kernel::argmax_kernel_cu(logits, size, stream);
        return next;
    }
#endif
    else {
        LOG(FATAL) << "Unsupported device type for ArgmaxSampler";
    }
}
}  // namespace sampler