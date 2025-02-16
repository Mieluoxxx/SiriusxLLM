#include "../src/op/kernels/interface.h"
#include "gtest/gtest.h"
#include "tensor/tensor.h"

// 定义一个测试用例，测试 embedding 算子在 CPU 上的行为
TEST(test_emb, test_cpu) {
    // 获取 CPU 设备的内存分配器实例
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    // 定义测试参数
    int32_t token = 4;   // token 的数量
    int32_t dim = 512;   // 每个 token 的 embedding 维度
    int32_t size = 2048; // weight 张量的总大小（token * dim）

    //  input，大小为 1，表示一个 token ID，我们的目标是从 weight 中取出这个 token ID 对应的行
    tensor::Tensor input(base::DataType::FP32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 1; // 初始化输入张量的第一个值为 1（模拟 token ID）

    // weight，形状为 [token, dim]
    tensor::Tensor weight(base::DataType::FP32, token, dim, true, alloc_cpu);
    // output，大小为 dim
    tensor::Tensor output(base::DataType::FP32, dim, true, alloc_cpu);

    // 初始化权重张量 weight，填充连续的值
    for (int i = 0; i < size; ++i) {
        weight.index<float>(i) = static_cast<float>(i); // 将 weight 的第 i 个值设置为 i
    }

    // 调用 embedding 算子的 CPU 内核函数
    kernel::get_embedding_kernel(base::DeviceType::CPU)(input, weight, output,
                                                        token, nullptr);

    // 验证输出张量 output 的值是否符合预期
    for (int i = 0; i < dim; ++i) {
        // 检查 output 的第 i 个值是否等于 512 + i
        // 因为 input 的 token ID 是 1，weight 的第 1 行（从 0 开始）的值是 512 到 1023
        ASSERT_EQ(output.index<float>(i), 512 + i);
    }
}