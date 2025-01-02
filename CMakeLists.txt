cmake_minimum_required(VERSION 3.20)
project(siriusx-infer CXX)

set(CXX_STANDARD_REQUIRED ON)
set(CXX_STANDARD 20)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

add_subdirectory(test)