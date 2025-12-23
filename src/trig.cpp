#include <cmath>
#include <omp.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif


extern "C" {
    
    // Sine
    EXPORT void prime_sin(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = std::sin(x[i]);
        }
    }

    // Cosine
    EXPORT void prime_cos(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = std::cos(x[i]);
        }
    }

    // Tangent
    EXPORT void prime_tan(long long n, double* res, double* x) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = std::tan(x[i]);
        }
    }
}
