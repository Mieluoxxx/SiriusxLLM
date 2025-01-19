/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 17:26:24
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-15 21:29:45
 * @FilePath: /SiriusX-infer/siriusx/include/base/base.h
 * @Description: 
 */
#ifndef BASE_H_
#define BASE_H_

#include <cstdint>
#include <cstddef>

namespace base {
enum class DeviceType: uint8_t {
    Unknown = 0,
    CPU = 1,
    CUDA = 2,
};

enum class DataType: uint8_t {
    Unknown = 0,
    FP32 = 1,
    Int8 = 2,
    Int32 = 3,
};

inline size_t DataTypeSize(DataType type) {
    switch (type) {
        case DataType::FP32:
            return sizeof(float);
        case DataType::Int8:
            return sizeof(int8_t);
        case DataType::Int32:
            return sizeof(int32_t);
        default:
            return 0;
    }
}

class NoCopyable { // 禁止拷贝构造和赋值
   protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};
}  // namespace base

#endif