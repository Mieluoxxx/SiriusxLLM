<!--
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 16:44:41
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-27 20:31:09
 * @FilePath: /siriusx-infer/README.md
 * @Description: 
-->
## 前置要求
cmake(v3.20)、vcpkg、g++/clang++(支持C++17)、ninja

## 启动命令

```bash
# 选择CUDA对应的gcc版本
export CXX=/usr/bin/g++-13

vcpkg new --application
vcpkg add port xxx
vcpkg x-update-baseline --add-initial-baseline 
```


## 好用的插件
`C++ TestMate`、`koroFileHeader`
```bash
# macos
ctrl+cmd+i 快速生成头部注释
ctrl+cmd+t 快速生成函数注释
# windows
ctrl+alt+i 快速生成头部注释
ctrl+alt+t 快速生成函数注释
```
`Todo Tree`


## 注意事项
CMakeLists.txt中需要添加`set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")`

vcpkg x-update-baseline --add-initial-baseline 

CUDA需要对应的gcc版本（ArchLinux需要注意）


## 碎碎念
`include/base/alloc.h`中蕴含的设计模式思想值得学习
`vcpkg`的glog默认是**静态库**，


## NVIDIA-Docker
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# cuda版本小于等于自己的驱动版本
docker pull nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
docker run --gpus all -t -i --name kuiperllama 5d846bce3f98 /bin/bash 

apt update
apt install -y vim net-tools openssh-server libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev wget cmake git gdb rsync
```