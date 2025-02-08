#!/usr/bin/env sh
###
 # @Author: Morgan Woods weiyiding0@gmail.com
 # @Date: 2025-02-01 06:47:05
 # @LastEditors: Morgan Woods weiyiding0@gmail.com
 # @LastEditTime: 2025-02-08 19:17:21
 # @FilePath: /siriusx-infer/platform.sh
 # @Description: 
### 

# 获取当前平台
current_platform=$(uname -s)

# 根据平台设置配置
case $current_platform in
    "Linux")
        settings='{
            "C_Cpp.default.cppStandard": "c++17",
            "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
            "cmake.configureArgs": [
                "-DCMAKE_TOOLCHAIN_FILE=/home/moguw/.vcpkg/scripts/buildsystems/vcpkg.cmake",
                "-DVCPKG_INSTALLED_DIR=/home/moguw/workspace/siriusx-infer/vcpkg_installed",
                "-DUSE_CUDA=ON"
            ],
            "cmake.generator": "Ninja"
        }'
        ;;
    "Darwin")
        settings='{
            "C_Cpp.default.cppStandard": "c++17",
            "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
            "cmake.configureArgs": [
                "-DVCPKG_INSTALLED_DIR=/Users/moguw/workspace/siriusx-infer/vcpkg_installed",
                "-DCMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja",
                "-DCMAKE_CXX_COMPILER=/usr/bin/c++"
            ],
            "cmake.generator": "Ninja"
        }'
        ;;
    "Windows"|"CYGWIN"*|"MINGW"*|"MSYS"*)
        settings='{
            "C_Cpp.default.cppStandard": "c++17",
            "cmake.configureArgs": [
                "-DCMAKE_TOOLCHAIN_FILE=C:\\path\\to\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake",
                "-DCMAKE_MAKE_PROGRAM=C:\\path\\to\\ninja.exe",
                "-DCMAKE_CUDA_COMPILER=C:\\path\\to\\nvcc.exe"
            ],
            "cmake.generator": "Ninja",
            "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
        }'
        ;;
    *)
        echo "Unsupported platform: $current_platform"
        exit 1
        ;;
esac

# 确保 .vscode 目录存在
vscode_dir=".vscode"
if [ ! -d "$vscode_dir" ]; then
    mkdir -p "$vscode_dir"
fi

# 生成 settings.json 文件
settings_path="$vscode_dir/settings.json"
echo "$settings" | jq . > "$settings_path"

echo "Generated $settings_path for $current_platform"