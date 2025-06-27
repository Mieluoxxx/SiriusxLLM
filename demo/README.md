# QWEN2.5 示例程序

本目录包含了 QWEN2.5 模型的各种示例程序，展示了模型的不同使用场景和功能。

## 示例程序列表

### 1. 文本生成 (generate_qwen2)
- 文件：`generate_qwen2.cpp`
- 功能：基础的文本生成示例，展示如何使用模型生成连续文本
- 特点：
  - 支持自定义生成长度
  - 支持 CPU/CUDA 加速
  - 可配置是否使用量化模型

### 2. 聊天功能 (chat_qwen2)
- 文件：`chat_qwen2.cpp`
- 功能：完整的聊天机器人实现，支持多轮对话
- 特点：
  - 使用 ChatML 格式
  - 支持系统提示词配置
  - 支持上下文管理
  - 实时输出生成内容

### 3. 性能测试 (benchmark_qwen2)
- 文件：`benchmark_qwen2.cpp`
- 功能：端到端推理性能与资源利用率分析
- 测试指标：
  - TTFT (Time to First Token)
  - TPOT (Time Per Output Token)
  - 吞吐量 (tokens/s)
  - 端到端延迟
  - 内存占用

## 编译说明

所有示例程序都需要开启 QWEN2.5 支持进行编译：

```bash
# 在项目根目录下
mkdir build && cd build
cmake -DQWEN2_SUPPORT=ON ..
make
```

## 使用说明

### 文本生成示例
```bash
./generate_qwen2 <模型路径> <分词器路径> <是否量化> <是否使用CUDA> [最大生成长度] [提示词]
```

### 聊天功能示例
```bash
./chat_qwen2 <模型路径> <分词器路径> <是否量化> <是否使用CUDA> [最大生成长度] [系统提示词]
```

### 性能测试示例
```bash
./benchmark_qwen2 <模型路径> <分词器路径>
``` 