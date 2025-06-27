cmake_minimum_required(VERSION 3.20)

# 设置编译命令导出
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 显示详细的构建命令，提供更多的调试信息
set(CMAKE_VERBOSE_MAKEFILE ON)

# 设置构建类型
set(CMAKE_BUILD_TYPE "Debug")

# 添加 USE_CUDA 选项，默认值为 OFF
option(USE_CUDA "Enable CUDA support" OFF)

# 添加 QWEN2_SUPPORT 选项，默认值为 OFF
option(QWEN2_SUPPORT "Enable Qwen2 support" OFF)

if(USE_CUDA)
    if(MSVC)
        # Setting this to true brakes Visual Studio builds.
        set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
    endif()
    # 符合我设备的 CUDA 版本制定编译期
    set(CMAKE_CUDA_COMPILER nvcc)
    # 设置 CUDA 标准
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
    
    # 设置支持的CUDA架构
    # 这里设置了较新的架构版本，如果需要支持更老的GPU，可以添加相应的架构
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    add_definitions(-DUSE_CUDA)
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译器标志
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fopenmp")
endif()

# 设置项目名称和语言
project(SiriusxLLM LANGUAGES CXX)

if(USE_CUDA)
    enable_language(CUDA)
    list(APPEND CMAKE_PROJECT_LANGUAGES CUDA)
endif()

# 添加第三方依赖
add_subdirectory(third_party/gflags)
add_subdirectory(third_party/googletest)
add_subdirectory(third_party/glog)
add_subdirectory(third_party/re2)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/json)
add_subdirectory(third_party/sentencepiece)
add_subdirectory(third_party/armadillo)

# 收集源代码文件
aux_source_directory(siriusx/src/base/ DIR_BASE)
aux_source_directory(siriusx/src/tensor/ DIR_TENSOR)
aux_source_directory(siriusx/src/op/ DIR_OP)
aux_source_directory(siriusx/src/op/kernels/ DIR_KERNELS)
aux_source_directory(siriusx/src/op/kernels/cpu/ DIR_KERNELS_CPU)
aux_source_directory(siriusx/src/op/kernels/cuda/ DIR_KERNELS_CUDA)
aux_source_directory(siriusx/src/sampler DIR_SMAPLER)
aux_source_directory(siriusx/src/model DIR_MODEL)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 添加库目标
if(USE_CUDA)
    add_library(llama STATIC ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_KERNELS} ${DIR_SMAPLER} ${DIR_MODEL} ${DIR_KERNELS_CPU} ${DIR_KERNELS_CUDA})
else()
    add_library(llama STATIC ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_KERNELS} ${DIR_SMAPLER} ${DIR_MODEL} ${DIR_KERNELS_CPU})
endif()

# 设置库的链接依赖
target_link_libraries(llama PUBLIC
    gtest_main
    gtest
    glog
    armadillo
    pthread
    sentencepiece
)

if(USE_CUDA)
    target_link_libraries(llama PUBLIC cudart)
    # 设置库的链接目录
    target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)
    # 设置 CUDA 可分离编译
    set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# 设置库的包含目录
target_include_directories(llama PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/glog/src
    ${PROJECT_SOURCE_DIR}/siriusx/include
    ${PROJECT_SOURCE_DIR}/third_party/armadillo/include
    ${PROJECT_SOURCE_DIR}/third_party/sentencepiece/src
    ${PROJECT_SOURCE_DIR}/siriusx/src
)

if(USE_CUDA)
    target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_definitions(-DUSE_CUDA)
endif()

if (QWEN2_SUPPORT)
    message(STATUS "QWEN2 SUPPORT")
    target_link_libraries(llama PUBLIC 
        absl::base 
        absl::strings
        re2::re2 
        nlohmann_json::nlohmann_json
    )
    add_definitions(-DQWEN2_SUPPORT)
endif()

# 添加子目录（例如测试目录）
add_subdirectory(test)
add_subdirectory(demo)