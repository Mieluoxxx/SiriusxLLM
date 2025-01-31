#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// CUDA 核函数：向量加法
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 测试用例
TEST(Example, test_cuda) {
    const int N = 1024;  // 向量大小
    const size_t size = N * sizeof(float);

    // 主机端数据
    std::vector<float> h_A(N, 1.0f);  // 初始化向量 A 为 1.0
    std::vector<float> h_B(N, 2.0f);  // 初始化向量 B 为 2.0
    std::vector<float> h_C(N, 0.0f);  // 结果向量 C 初始化为 0.0

    // 设备端数据
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 定义 CUDA 网格和块的尺寸
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动 CUDA 核函数
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(h_C[i], 3.0f);  // 1.0 + 2.0 = 3.0
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}