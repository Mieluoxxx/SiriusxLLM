#ifndef SAMPLER_H
#define SAMPLER_H

#include "base/base.h"

// clang-format off
namespace sampler {
class Sampler {
   public:
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}
    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

   protected:
    base::DeviceType device_type_;
};
}  // namespace sampler

#endif