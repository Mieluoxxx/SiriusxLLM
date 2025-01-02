<!--
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-02 16:44:41
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-02 20:35:15
 * @FilePath: /SiriusX-infer/README.md
 * @Description: 
-->
## 启动命令

```bash
vcpkg new --application
vcpkg add port catch2
```


## 额外插件
`C++ TestMate`

`koroFileHeader`
```bash
# macos
ctrl+cmd+i 快速生成头部注释
ctrl+cmd+t 快速生成函数注释
# windows
ctrl+alt+i 快速生成头部注释
ctrl+alt+t 快速生成函数注释
```

## 注意事项
CMakeLists.txt中需要添加`set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")`