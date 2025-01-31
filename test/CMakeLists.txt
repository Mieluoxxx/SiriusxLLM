include(../cmake/cuda.cmake)
find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)

aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/test_example DIR_TEST_EXAMPLE)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/test_op DIR_TEST_OP)


add_executable(my_test
    ${DIR_TEST}
    ${DIR_TEST_EXAMPLE}
    ${DIR_TEST_TENSOR}
    ${DIR_TEST_OP}
)

target_link_libraries(my_test PUBLIC GTest::gtest_main GTest::gtest glog::glog llama)
target_include_directories(my_test PUBLIC
    ${glog_INCLUDE_DIR}
    ${GTest_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/siriusx/include
)

set_target_properties(my_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)