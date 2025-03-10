/***
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 19:09:35
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-02-01 02:25:17
 * @FilePath: /SiriusxLLM/test/test_example/example.cpp
 * @Description:
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

int add(const int a, const int b) { return a + b; }

TEST(Example, AdditionTest) {
    int a = 1;
    int b = 2;
    int c = add(a, b);
    EXPECT_EQ(c, 3);
    LOG(INFO) << "Example.AdditionTest passed.";
}