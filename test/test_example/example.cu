#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <gtest/gtest.h>
// 简单的 CUDA 核函数，将数组中的每个元素乘以 2
__global__ void multiplyByTwo(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2;
    }
}

// CUDA 测试案例
TEST(Example, CudaTest) {
    const int size = 10;
    int h_data[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // 主机端数据
    int* d_data;                                         // 设备端数据指针

    // 分配设备端内存
    cudaMalloc((void**)&d_data, size * sizeof(int));

    // 将数据从主机端复制到设备端
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 CUDA 核函数
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    multiplyByTwo<<<gridSize, blockSize>>>(d_data, size);

    // 将结果从设备端复制回主机端
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(h_data[i], (i + 1) * 2);  // 每个元素应该乘以 2
    }

    // 释放设备端内存
    cudaFree(d_data);
}
#endif