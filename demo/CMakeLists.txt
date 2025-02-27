add_executable(llama2 main_llama2.cpp)
target_link_libraries(llama2 PUBLIC llama)

target_include_directories(my_test PUBLIC
    ${PROJECT_SOURCE_DIR}/siriusx/include
)