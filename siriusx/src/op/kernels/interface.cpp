/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-26 15:34:19
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-26 18:08:14
 * @FilePath: /SiriusX-infer/siriusx/src/op/kernels/interface.cpp
 * @Description: 
 */
#include "interface.h"

#include <base/base.h>

#include "cpu/add_kernel.h"

namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::CPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::CUDA) {
        LOG(FATAL) << "CUDA is not supported yet";  // TODO : add cuda kernel
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}
}  // namespace kernel