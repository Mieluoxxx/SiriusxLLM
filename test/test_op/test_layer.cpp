#include <gtest/gtest.h>

#include "base/base.h"
#include "op/layer.h"

TEST(Layer, ConstructionAndBasicFunction) {
    using namespace op;
    Layer layer(base::DeviceType::CPU, LayerType::Unknown, "Layer");
    // Layer层默认的数据类型为FP32
    EXPECT_EQ(layer.data_type(), base::DataType::FP32);
    EXPECT_EQ(layer.layer_type(), LayerType::Unknown);
    EXPECT_EQ(layer.get_layer_name(), "Layer");
    EXPECT_EQ(layer.device_type(), base::DeviceType::CPU);

    layer.set_layer_name("NewLayer");
    EXPECT_EQ(layer.get_layer_name(), "NewLayer");

    layer.set_device_type(base::DeviceType::CUDA);
    EXPECT_EQ(layer.device_type(), base::DeviceType::CUDA);
}

TEST(LayerParam, ConstructionAndBasicFunction) {
    using namespace op;
    LayerParam layer_param(base::DeviceType::CPU, LayerType::Unknown, false,
                           "LayerParam");
    EXPECT_EQ(layer_param.data_type(), base::DataType::FP32);
    EXPECT_EQ(layer_param.layer_type(), LayerType::Unknown);
    EXPECT_EQ(layer_param.get_layer_name(), "LayerParam");
    EXPECT_EQ(layer_param.device_type(), base::DeviceType::CPU);

    layer_param.set_layer_name("NewLayerParam");
    EXPECT_EQ(layer_param.get_layer_name(), "NewLayerParam");

    layer_param.set_device_type(base::DeviceType::CUDA);
    EXPECT_EQ(layer_param.device_type(), base::DeviceType::CUDA);
}
