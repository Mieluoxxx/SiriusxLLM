# 默认编译llama2
if(NOT QWEN2_SUPPORT AND NOT LLAMA3_SUPPORT)
    add_executable(raw demo_llama2.cpp)
    target_link_libraries(raw PUBLIC llama)
    add_executable(llama2 gen_llama2.cpp)
    target_link_libraries(llama2 PUBLIC llama)
endif()

# 如果开启llama3，编译llama3
if(LLAMA3_SUPPORT)
    add_executable(raw demo_llama2.cpp)
    target_link_libraries(raw PUBLIC llama)
    add_executable(llama2 gen_llama2.cpp)
    target_link_libraries(llama2 PUBLIC llama)
    
    add_executable(llama3 demo_llama3.cpp)
    target_link_libraries(llama3 PUBLIC llama)
endif()

# 如果开启qwen2，只编译qwen2
if(QWEN2_SUPPORT)
    add_executable(qwen demo_qwen2.cpp)
    target_link_libraries(qwen PUBLIC llama)
    add_executable(qwen_gen gen_qwen2.cpp)
    target_link_libraries(qwen_gen PUBLIC llama)   
    add_executable(chat demo_chat_qwen2.cpp)
    target_link_libraries(chat PUBLIC llama)
    add_executable(qwen_chat chat_qwen2.cpp)
    target_link_libraries(qwen_chat PUBLIC llama)
endif()
