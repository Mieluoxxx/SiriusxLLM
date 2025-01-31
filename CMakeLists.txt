cmake_minimum_required(VERSION 3.20)

# 设置编译命令导出
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 显示详细的构建命令，提供更多的调试信息
set(CMAKE_VERBOSE_MAKEFILE ON)

# 设置构建类型
set(CMAKE_BUILD_TYPE "Debug")

# 设置项目名称和语言
project(siriusx-infer LANGUAGES CXX CUDA)

# 设置工具链文件（如果使用 vcpkg）
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

include(cmake/cuda.cmake)
# 设置 CUDA 编译器
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)

# 收集源代码文件
aux_source_directory(siriusx/src/base/ DIR_BASE)
aux_source_directory(siriusx/src/tensor/ DIR_TENSOR)
aux_source_directory(siriusx/src/op/ DIR_OP)
aux_source_directory(siriusx/src/op/kernels/ DIR_KERNELS)
aux_source_directory(siriusx/src/op/kernels/cpu/ DIR_KERNELS_CPU)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 添加库目标
add_library(llama STATIC ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_KERNELS} ${DIR_KERNELS_CPU})

# 设置库的链接依赖
target_link_libraries(llama PUBLIC
    GTest::gtest_main
    GTest::gtest
    glog::glog
    armadillo
    pthread
    cudart
)

# 设置库的链接目录
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# 设置库的包含目录
target_include_directories(llama PUBLIC
    ${glog_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/siriusx/include
    ${Armadillo_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 设置 CUDA 可分离编译
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 添加子目录（例如测试目录）
add_subdirectory(test)
