/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 17:26:24
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-04 18:19:07
 * @FilePath: /SiriusX-infer/siriusx/include/base/base.h
 * @Description: 
 */
#ifndef BASE_H_
#define BASE_H_

namespace base {
enum class DeviceType {
    Unknown = 0,
    CPU = 1,
    CUDA = 2,
};

class NoCopyable {
   protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};
}  // namespace base

#endif