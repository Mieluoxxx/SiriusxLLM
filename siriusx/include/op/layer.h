/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-15 21:14:01
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-24 19:11:26
 * @FilePath: /SiriusxLLM/siriusx/include/op/layer.h
 * @Description:
 */
#ifndef OP_LAYER_H
#define OP_LAYER_H

#include <cstdint>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace op {
enum LayerType : uint8_t {
    Unknown = 0,
    LayerAdd = 1,
    LayerMatmul = 2,
    LayerRMSNorm = 3,
    LayerEmbedding = 4,
    LayerSwiGLU = 5,
    LayerRoPE = 6,
    LayerMHA = 7,
    LayerEncode = 8,
};

// clang-format off
class BaseLayer {
   public:
    // 构造函数，初始化BaseLayer对象
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name = "");

    // 获取数据类型
    base::DataType data_type() const;

    // 获取层类型
    LayerType layer_type() const;

    // 初始化函数，纯虚函数，需要在子类中实现
    virtual base::Status init() = 0;

    // 前向传播函数，纯虚函数，需要在子类中实现
    virtual base::Status forward() = 0;
    virtual base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& out) = 0;
    virtual base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2,
                                 const tensor::Tensor& out) = 0;
    virtual base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2, const tensor::Tensor& in3,
                                 const tensor::Tensor& out) = 0;
    virtual base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2, const tensor::Tensor& in3,
                                 const tensor::Tensor& in4, const tensor::Tensor& out) = 0;
    virtual base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2, const tensor::Tensor& in3,
                                 const tensor::Tensor& in4, const tensor::Tensor& in5, const tensor::Tensor& out) = 0;

    // 设置输入
    virtual void set_input(int32_t idx, const tensor::Tensor& in) = 0;
    // 设置输出
    virtual void set_output(int32_t idx, const tensor::Tensor& out) = 0;

    // 获取输入大小
    virtual size_t input_size() const = 0;
    // 获取输出大小
    virtual size_t output_size() const = 0;

    // 检查函数，纯虚函数，需要在子类中实现
    virtual base::Status check() const = 0;

    // 获取输入
    virtual tensor::Tensor& get_input(int32_t idx) = 0;
    // 获取输出
    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    // 获取输入（const）
    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;
    // 获取输出（const）
    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    // 设置权重
    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);
    virtual base::Status set_weight(
        int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::Unknown);

    // 获取层名称
    const std::string& get_layer_name() const;

    // 设置层名称
    void set_layer_name(const std::string& layer_name);

    // 获取设备类型
    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

   protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::Unknown;
    base::DataType data_type_ = base::DataType::Unknown;
    base::DeviceType device_type_ = base::DeviceType::Unknown;
};

class Layer : public BaseLayer {
   public:
    // 构造函数，初始化Layer对象
    explicit Layer(base::DeviceType device_type, LayerType layer_type,
                   std::string layer_name = "");

    // 初始化Layer对象
    base::Status init() override;

    // 检查tensor是否合法
    base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                              base::DataType data_type) const;

    // 检查tensor是否合法，并传入可变参数
    base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                       base::DataType data_type, ...) const;

    // 检查Layer对象是否合法
    base::Status check() const override;

    // 前向传播
    base::Status forward() override;
    base::Status forward(const tensor::Tensor& in1,
                         const tensor::Tensor& out1) override;
    base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2,
                         const tensor::Tensor& out1) override;
    base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2,
                         const tensor::Tensor& in3,
                         const tensor::Tensor& out1) override;
    base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2,
                         const tensor::Tensor& in3, const tensor::Tensor& in4,
                         const tensor::Tensor& out1) override;
    base::Status forward(const tensor::Tensor& in1, const tensor::Tensor& in2,
                         const tensor::Tensor& in3, const tensor::Tensor& in4,
                         const tensor::Tensor& in5,
                         const tensor::Tensor& out1) override;

    // 设置输入tensor
    virtual void set_input(int32_t idx, const tensor::Tensor& in) override;
    // 设置输出tensor
    virtual void set_output(int32_t idx, const tensor::Tensor& out) override;

    // 获取输入tensor
    tensor::Tensor& get_input(int32_t idx) override;
    // 获取输出tensor
    tensor::Tensor& get_output(int32_t idx) override;

    // 获取输入tensor
    const tensor::Tensor& get_input(int32_t idx) const override;
    // 获取输出tensor
    const tensor::Tensor& get_output(int32_t idx) const override;

    size_t input_size() const override;
    size_t output_size() const override;

    void reset_input_size(size_t size);
    void reset_output_size(size_t size);

    virtual void to_cuda();
    void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);
    std::shared_ptr<kernel::CudaConfig> cuda_config() const;

   protected:
    std::vector<tensor::Tensor> inputs_;
    std::vector<tensor::Tensor> outputs_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

class LayerParam : public Layer {
   public:
    // 构造函数，初始化LayerParam对象
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                        bool is_quant_layer = false,
                        std::string layer_name = "");

    // 获取权重大小
    size_t weight_size() const;

    // 重置权重大小
    void reset_weight_size(size_t size);

    // 获取指定索引的权重
    tensor::Tensor& get_weight(int32_t idx);

    // 获取指定索引的权重（常量）
    const tensor::Tensor& get_weight(int32_t idx) const;

    void to_cuda() override;

    // 设置指定索引的权重
    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

    // 设置指定索引的权重（通过维度和指针）
    base::Status set_weight(
        int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::Unknown) override;

    // 设置缩放因子
    void set_scales(const tensor::Tensor& scales);

    // 设置分组大小
    void set_group_size(int32_t group_size);

    // 获取缩放因子数量
    int32_t get_scale_num() const;

   protected:
    // 分组大小
    int32_t group_size_ = 0;
    // 是否为量化层
    bool is_quant_layer_ = false;
    // 缩放因子
    tensor::Tensor scales_;
    // 权重
    std::vector<tensor::Tensor> weights_;
};

}  // namespace op

#endif  // OP_LAYER_H