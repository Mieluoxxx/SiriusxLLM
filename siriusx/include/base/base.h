/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 17:26:24
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-23 07:04:57
 * @FilePath: /SiriusX-infer/siriusx/include/base/base.h
 * @Description:
 */
#ifndef BASE_H_
#define BASE_H_

#include <cstdint>
#include <string>

#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)

namespace base {
enum class DeviceType : uint8_t {
    Unknown = 0,
    CPU = 1,
    CUDA = 2,
};

enum class DataType : uint8_t {
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

class NoCopyable {  // 禁止拷贝构造和赋值
   protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode {
    Success = 0,
    FunctionUnImplement = 1,
    InvalidArgument = 2,
};

class Status {
   public:
    // 构造函数，默认值为StatusCode::Success
    Status(int code = StatusCode::Success, std::string err_msg = "");
    // 拷贝构造函数
    Status(const Status& other) = default;
    // 赋值运算符重载
    Status& operator=(const Status& other) = default;
    // 赋值运算符重载，将int类型的code赋值给Status对象
    Status& operator=(int code);

    // 比较运算符重载
    bool operator==(int code) const;
    bool operator!=(int code) const;

    // 类型转换运算符重载
    operator int() const;
    operator bool() const;

    // 获取 Status 对象的 code
    int32_t get_err_code() const;

    // 获取 Status 对象的 err_msg
    const std::string& get_err_msg() const;

    // 设置 Status 对象的 err_msg
    void set_err_msg(const std::string& err_msg);

   private:
    int code_ = StatusCode::Success;
    std::string msg_;
};

namespace error {
#define STATUS_CHECK(call)                                                    \
    do {                                                                      \
        const base::Status& status = call;                                    \
        if (!status) {                                                        \
            const size_t buf_size = 512;                                      \
            char buf[buf_size];                                               \
            snprintf(buf, buf_size - 1,                                       \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error " \
                     "msg:%s\n",                                              \
                     __FILE__, __LINE__, int(status),                         \
                     status.get_err_msg().c_str());                           \
            LOG(FATAL) << buf;                                                \
        }                                                                     \
    } while (0)

Status Success(const std::string& err_msg = "");
Status FunctionNotImplemented(const std::string& err_msg = "");
Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

}  // namespace base

#endif