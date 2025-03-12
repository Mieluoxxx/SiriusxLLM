#include "emb_kernel.h"

#include "base/alloc.h"
#include "tensor/tensor.h"
namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size,
                       void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);
    CHECK(weight.device_type() == output.device_type());
    CHECK(input.device_type() == base::DeviceType::CPU);

    const auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    for (int32_t i = 0; i < input_num; ++i) {
        int32_t token = *input.ptr<int32_t>(i);
        if (token > vocab_size) {
            LOG(FATAL) << "token " << token << " is out of range";
        } else {
            float* dest_ptr =
                const_cast<float*>(output.ptr<float>(i * weight_dim));
            float* src_ptr =
                const_cast<float*>(weight.ptr<float>(token * weight_dim));
            if (weight.device_type() == base::DeviceType::CPU) {
                alloc->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float),
                              base::MemcpyKind::CPU2CPU);
            } else {
                LOG(FATAL) << "Unknown device type of weight tensor in the "
                              "embedding layer.";
            }
        }
    }
}
}  // namespace kernel