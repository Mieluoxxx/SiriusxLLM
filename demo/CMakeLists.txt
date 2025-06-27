# 默认编译llama2
if(NOT QWEN2_SUPPORT)
    add_executable(llama2 llama2.cpp)
    target_link_libraries(llama2 PUBLIC llama)
endif()

# 编译 QWEN2.5 相关的示例程序
if(QWEN2_SUPPORT)
    # 基础文本生成示例
    add_executable(generate_qwen2 generate_qwen2.cpp)
    target_link_libraries(generate_qwen2 PUBLIC llama)

    # 聊天功能示例
    add_executable(chat_qwen2 chat_qwen2.cpp)
    target_link_libraries(chat_qwen2 PUBLIC llama)

    # 性能测试示例
    add_executable(benchmark_qwen2 benchmark_qwen2.cpp)
    target_link_libraries(benchmark_qwen2 PUBLIC llama)
endif()
