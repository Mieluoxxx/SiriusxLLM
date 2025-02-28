/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-24 20:39:57
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 21:46:42
 * @FilePath: /siriusx-infer/siriusx/include/sampler/argmax_sampler.h
 * @Description: 
 */
#ifndef ARGMAX_SAMPLER_H
#define ARGMAX_SAMPLER_H

#include "base/base.h"
#include "sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
   public:
    explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}
    size_t sample(const float* logits, size_t size, void* stream) override;
};
}  // namespace sampler

#endif