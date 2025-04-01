add_executable(raw generate.cpp)
target_link_libraries(raw PUBLIC llama)

add_executable(llama2 main_llama2.cpp)
target_link_libraries(llama2 PUBLIC llama)

add_executable(chat chat.cpp)
target_link_libraries(chat PUBLIC llama)

if (QWEN2_SUPPORT)
add_executable(qwen main_qwen2.cpp)
target_link_libraries(qwen PUBLIC llama)
endif()

if (LLAMA3_SUPPORT)
add_executable(llama3 main_llama3.cpp)
target_link_libraries(llama3 PUBLIC llama)
endif()