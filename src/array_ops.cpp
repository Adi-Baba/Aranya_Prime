#include <omp.h>

extern "C" {
    
    // Subtraction: res = a - b
    __declspec(dllexport) void prime_sub(long long n, double* res, double* a, double* b) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = a[i] - b[i];
        }
    }

    // Multiplication: res = a * b
    __declspec(dllexport) void prime_mul(long long n, double* res, double* a, double* b) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = a[i] * b[i];
        }
    }

    // Division: res = a / b
    __declspec(dllexport) void prime_div(long long n, double* res, double* a, double* b) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = a[i] / b[i];
        }
    }
}
