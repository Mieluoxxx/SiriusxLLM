#include <catch2/catch_test_macros.hpp>

int add(const int a, const int b) {
    return a + b;
}

TEST_CASE("Test", "[add]") {
    REQUIRE(add(1, 2) == 3);
}