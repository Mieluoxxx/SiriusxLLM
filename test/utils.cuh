#ifdef USE_CUDA
#ifndef TEST_CU_UTILS
#define TEST_CU_UTILS

#include <cstdint>
void test_function(float* ptr, int32_t size, float value = 1.f);

void set_value_cu(float* arr_cu, int32_t size, float value = 1.f);

#endif  // TEST_CU_UTILS
#endif  // USE_CUDA