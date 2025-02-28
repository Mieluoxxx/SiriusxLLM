/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-27 20:48:55
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 18:30:57
 * @FilePath: /siriusx-infer/siriusx/include/sampler/sampler.h
 * @Description: 审查完成 0228
 */
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