find_package(Catch2 CONFIG REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test PUBLIC Catch2::Catch2 Catch2::Catch2WithMain)