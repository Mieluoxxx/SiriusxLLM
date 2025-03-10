#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <memory>

#include "base/alloc.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"

namespace base {
// 构造函数，初始化CUDADeviceAllocator对象
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::CUDA) {
}  // 调用父类DeviceAllocator的构造函数，传入设备类型为CUDA

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    // 获取设备ID
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    // 检查获取设备ID是否成功
    CHECK(state == cudaSuccess) << "Failed to get device id: " << cudaGetErrorString(state);

    // 如果需要分配的内存大于1MB
    if (byte_size > 1024 * 1024) {
        // 获取大内存缓冲区
        auto& big_buffers = big_buffers_map_[id];
        // 选择合适的缓冲区
        int sel_id = -1;
        for (int i = 0; i < big_buffers.size(); ++i) {
            if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
                    sel_id = i;
                }
            }
        }
        // 如果找到合适的缓冲区
        if (sel_id != -1) {
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }

        // 如果没有找到合适的缓冲区，则分配新的内存
        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (cudaSuccess != state) {
            char buf[256];
            snprintf(buf, 256,
                     "ERROR: CUDA error when allocating %lu MB memeory!",
                     byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }

    // 获取小内存缓冲区
    auto& cuda_buffers = cuda_buffers_map_[id];
    // 选择合适的缓冲区
    for (int i = 0; i < cuda_buffers.size(); ++i) {
        if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
            cuda_buffers[i].busy = true;
            no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
            return cuda_buffers[i].data;
        }
    }

    // 如果没有找到合适的缓冲区，则分配新的内存
    void* ptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
        char buf[256];
        snprintf(buf, 256, "ERROR: CUDA error when allocating %lu MB memeory!",
                 byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    // 如果指针为空，则直接返回
    if (!ptr) return;
    // 如果cuda_buffers_map_为空，则直接返回
    if (cuda_buffers_map_.empty()) return;
    cudaError_t state = cudaSuccess;
    // 遍历cuda_buffers_map_中的每个设备
    for (auto& it : cuda_buffers_map_) {
        // 如果该设备的空闲次数超过2^30次，则释放该设备的所有内存
        if (no_busy_cnt_[it.first] > 2 >> 30) {
            auto& cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            // 遍历该设备的所有内存
            for (int i = 0; i < cuda_buffers.size(); i++) {
                // 如果该内存块没有被占用，则释放该内存块
                if (!cuda_buffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cuda_buffers[i].data);
                    // 检查释放内存是否成功
                    CHECK(state == cudaSuccess)
                        << "Error: CUDA error when release memory on device"
                        << it.first;
                } else {
                    // 如果该内存块被占用，则将其添加到临时向量中
                    temp.push_back(cuda_buffers[i]);
                }
            }
            // 清空该设备的内存向量
            cuda_buffers.clear();
            // 将临时向量赋值给该设备的内存向量
            it.second = temp;
            // 重置该设备的空闲次数
            no_busy_cnt_[it.first] = 0;
        }
    }

    // 再次遍历cuda_buffers_map_中的每个设备
    for (auto& it : cuda_buffers_map_) {
        auto& cuda_buffers = it.second;
        // 遍历该设备的所有内存
        for (int i = 0; i < cuda_buffers.size(); i++) {
            // 如果该内存块的数据指针与要释放的指针相同，则将该内存块的空闲次数增加其字节大小，并将该内存块标记为空闲
            if (cuda_buffers[i].data == ptr) {
                no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                cuda_buffers[i].busy = false;
                return;
            }
        }
        // 如果该内存块不在该设备的内存向量中，则遍历该设备的big_buffers_map_中的所有内存
        auto& big_buffers = big_buffers_map_[it.first];
        for (int i = 0; i < big_buffers.size(); i++) {
            // 如果该内存块的数据指针与要释放的指针相同，则将该内存块标记为空闲
            if (big_buffers[i].data == ptr) {
                big_buffers[i].busy = false;
                return;
            }
        }
    }
    // 如果要释放的指针不在任何设备的内存向量中，则直接释放该指针
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory";
}

// clang-format off
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base

#endif  // USE_CUDA