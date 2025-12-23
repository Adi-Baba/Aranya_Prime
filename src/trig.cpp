#include <cmath>
#include <omp.h>

extern "C" {
    
    // Sine
    __declspec(dllexport) void prime_sin(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = std::sin(x[i]);
        }
    }

    // Cosine
    __declspec(dllexport) void prime_cos(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = std::cos(x[i]);
        }
    }

    // Tangent
    __declspec(dllexport) void prime_tan(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = std::tan(x[i]);
        }
    }
}
