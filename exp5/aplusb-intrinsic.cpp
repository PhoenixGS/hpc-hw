#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8)
    {
        __m256 m1 = _mm256_load_ps(a + i);
        __m256 m2 = _mm256_load_ps(b + i);
        __m256 m = _mm256_add_ps(m1, m2);
        _mm256_store_ps(c + i, m);
    }
}