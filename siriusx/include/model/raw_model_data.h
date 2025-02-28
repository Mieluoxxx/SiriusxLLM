/*
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-02-27 20:48:55
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-28 17:51:01
 * @FilePath: /siriusx-infer/siriusx/include/model/raw_model_data.h
 * @Description: 审查完成
 */

#ifndef RAW_MODEL_DATA_H
#define RAW_MODEL_DATA_H

#include <cstddef>
#include <cstdint>
namespace model {
struct RawModelData {
    ~RawModelData();
    int32_t fd = -1;
    size_t file_size = 0;
    void* data = nullptr;
    void* weight_data = nullptr;

    virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFP32 : RawModelData {
    const void* weight(size_t offset) const override;
};

struct RawModelDataINT8 : RawModelData {
    const void* weight(size_t offset) const override;
};
}  // namespace model

#endif  // RAW_MODEL_DATA_H