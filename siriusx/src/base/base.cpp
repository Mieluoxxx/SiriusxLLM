/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 17:26:29
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-26 12:56:05
 * @FilePath: /SiriusX-infer/siriusx/src/base/base.cpp
 * @Description:
 */
#include "base/base.h"

#include <string>
namespace base {
Status::Status(int code, std::string err_msg)
    : code_(code), msg_(std::move(err_msg)) {}

Status& Status::operator=(int code) {
    code_ = code;
    return *this;
};

bool Status::operator==(int code) const {
    if (code_ == code) {
        return true;
    } else {
        return false;
    }
}

bool Status::operator!=(int code) const {
    if (code_ != code) {
        return true;
    } else {
        return false;
    }
};

Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == StatusCode::Success; }

int32_t Status::get_err_code() const { return code_; }

const std::string& Status::get_err_msg() const { return msg_; }

void Status::set_err_msg(const std::string& err_msg) { msg_ = err_msg; }

namespace error {
Status Success(const std::string& err_msg) {
    return Status{StatusCode::Success, err_msg};
}

Status FunctionNotImplemented(const std::string& err_msg) {
    return Status{StatusCode::FunctionUnImplement, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
    return Status{StatusCode::InvalidArgument, err_msg};
}
}  // namespace error
}  // namespace base