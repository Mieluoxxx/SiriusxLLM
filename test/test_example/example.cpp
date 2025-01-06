/*** 
 * @Author: Morgan Woods weiyiding0@gmail.com
 * @Date: 2025-01-04 19:09:35
 * @LastEditors: Morgan Woods weiyiding0@gmail.com
 * @LastEditTime: 2025-01-06 13:31:08
 * @FilePath: /SiriusX-infer/test/test_example/example.cpp
 * @Description: 
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

int add(const int a, const int b) { return a + b; }

TEST(Example, AdditionTest) {
    EXPECT_EQ(add(1, 2), 3);
    LOG(INFO) << "AdditionTest.BasicTest passed.";
}