#include <omp.h>

extern "C" {
    // x^3 + x^2 + x
    __declspec(dllexport) void prime_poly(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            double val = x[i];
            res[i] = (val * val * val) + (val * val) + val;
        }
    }
}
