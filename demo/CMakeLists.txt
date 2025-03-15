add_executable(llama2 main_llama2.cpp)
target_link_libraries(llama2 PUBLIC llama)

add_executable(raw_llama llama2.cpp)
target_link_libraries(raw_llama PUBLIC llama)