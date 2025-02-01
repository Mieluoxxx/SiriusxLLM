# 查找依赖包
find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)

# 收集测试源代码文件
aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/test_example DIR_TEST_EXAMPLE)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/test_op DIR_TEST_OP)

# 添加可执行目标
add_executable(my_test
    ${DIR_TEST}
    ${DIR_TEST_EXAMPLE}
    ${DIR_TEST_TENSOR}
    ${DIR_TEST_OP}
)

# 设置目标链接库
target_link_libraries(my_test PUBLIC
    GTest::gtest_main
    GTest::gtest
    glog::glog
    llama
)

# 设置目标包含目录
target_include_directories(my_test PUBLIC
    ${glog_INCLUDE_DIR}
    ${GTest_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/siriusx/include
)
