<!--
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 16:44:41
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-06-27 16:22:57
 * @FilePath: /siriusxllm/README.md
 * @Description: SiriusxLLM - 高性能大语言模型推理框架
-->
# SiriusxLLM

SiriusxLLM 是一个高性能的大语言模型推理框架，支持多种模型并提供 CPU 和 CUDA 加速。

## 功能特点

- 支持 QWEN2.5 等主流大语言模型
- 提供 CPU 和 CUDA 双重加速支持
- 高效的内存分配器设计
- 完整的单元测试覆盖

## 系统要求

- cmake (>= v3.20)
- g++/clang++ (支持C++17)
- ninja (可选，推荐)
- CUDA (可选)

## 快速开始

### 克隆项目
```bash
# 克隆项目及其所有子模块
git clone --recursive https://github.com/your-repository/siriusxllm.git

# 如果已经克隆但没有子模块，执行：
git submodule update --init --recursive
```

### 构建项目
```bash
# 创建并进入构建目录
mkdir build && cd build

# 配置项目（使用Ninja构建系统，推荐）
cmake -GNinja -DUSE_CUDA=ON -DQWEN2_SUPPORT=ON ..
ninja

# 或使用Make构建系统
cmake -DUSE_CUDA=ON -DQWEN2_SUPPORT=ON ..
make -j$(nproc)
```

## 项目结构
```
siriusxllm/
├── demo/          # 示例代码
├── siriusx/       # 核心库代码
├── test/          # 测试代码
├── tools/         # 工具和脚本
└── third_party/   # 第三方依赖
```

## 注意事项

- 确保编译器支持 C++17
- 使用 CUDA 时，请确保 GCC 版本与 CUDA 兼容
- 项目使用 git submodules 管理依赖，克隆时需包含 `--recursive` 参数
- 项目量化部分尚未完成