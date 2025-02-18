#include <gtest/gtest.h>
#include <glog/logging.h> 

#include <cmath>
#include <random>

#include "../src/op/kernels/interface.h"
#include "../src/op/kernels/cpu/rope_kernel.h"
#include "base/alloc.h"

namespace kernel {
    TEST(test_rope, test_cpu) {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    
        int32_t dim = 256;
        int32_t head_size = 64;
        int32_t kv_dim = 128;
        int32_t pos = 3;
        int32_t max_seq_len = 1024;  // 假设最大序列长度为1024
    
        // 创建位置张量
        tensor::Tensor input_pos(base::DataType::Int32, 1, true, alloc);
        input_pos.index<int32_t>(0) = pos;
    
        // 创建随机数生成器
        // std::random_device rd;
        // std::mt19937 mt(rd());
        std::mt19937 mt(42);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
    
        // 创建query和key张量
        tensor::Tensor input_q(base::DataType::FP32, dim, true, alloc);
        tensor::Tensor input_k(base::DataType::FP32, dim, true, alloc);
    
        // 初始化query和key张量
        for (int i = 0; i < dim; ++i) {
            input_q.index<float>(i) = dist(mt);
            input_k.index<float>(i) = dist(mt);
        }
    
        // 创建sin_cache和cos_cache张量
        tensor::Tensor sin_cache(base::DataType::FP32, max_seq_len * head_size, true, alloc);
        tensor::Tensor cos_cache(base::DataType::FP32, max_seq_len * head_size, true, alloc);
    
        // 计算sin_cache和cos_cache
        sin_cos_cache_calc_cpu(head_size, max_seq_len, sin_cache.ptr<float>(), cos_cache.ptr<float>());
    
        // 调用rope_kernel_cpu函数
        rope_kernel_cpu(dim, kv_dim, head_size, input_q, input_k, input_pos, sin_cache, cos_cache, nullptr);

        for(int i = 0; i < 3; i++) {
            LOG(INFO) << input_q.index<float>(i) << " ";
        }
        for(int i = 0; i < 3; i++) {
            LOG(INFO) << input_k.index<float>(i) << " ";
        }
    }
}  // namespace kernel