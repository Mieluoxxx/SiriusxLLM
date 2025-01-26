cmake_minimum_required(VERSION 3.20)
# 设置编译命令导出
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 显示详细的构建命令、提供更多的调试信息、帮助理解构建过程
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_BUILD_TYPE "Debug")


project(siriusx-infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------
find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
# ------------------------------------
aux_source_directory(siriusx/src/base/ DIR_BASE)
aux_source_directory(siriusx/src/tensor/ DIR_TENSOR)
aux_source_directory(siriusx/src/op/ DIR_OP)
aux_source_directory(siriusx/src/op/kernels/ DIR_KERNELS)
aux_source_directory(siriusx/src/op/kernels/cpu/ DIR_KERNELS_CPU)

# ------------------------------------
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ------------------------------------
add_library(llama STATIC ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_KERNELS} ${DIR_KERNELS_CPU})
target_link_libraries(llama PUBLIC GTest::gtest_main GTest::gtest glog::glog armadillo pthread)
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/siriusx/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})

add_subdirectory(test)