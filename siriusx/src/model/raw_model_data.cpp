/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-27 20:48:55
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 18:53:47
 * @FilePath: /siriusx-infer/siriusx/src/model/raw_model_data.cpp
 * @Description: 审察完成 0228
 */
#include "model/raw_model_data.h"

#include <sys/mman.h>
#include <unistd.h>

namespace model {
// 析构函数，释放资源
RawModelData::~RawModelData() {
    // 如果data不为空且不为MAP_FAILED，则释放内存
    if (data != nullptr && data != MAP_FAILED) {
        munmap(data, file_size);
        data = nullptr;
    }
    // 如果fd不为-1，则关闭文件描述符
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
}

const void* RawModelDataFP32::weight(size_t offset) const {
    return static_cast<float*>(weight_data) + offset;
}

const void* RawModelDataINT8::weight(size_t offset) const {
    return static_cast<int8_t*>(weight_data) + offset;
}
}  // namespace model