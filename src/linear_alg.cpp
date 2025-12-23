#include <cmath>
#include <omp.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

extern "C" {
    
    // Dot Product: sum(a * b)
    // Note: Returns a scalar via pointer to support ctypes easy handling or we can return double directly.
    // For consistency with arrays, let's write to a result pointer (size 1).
    // Dot Product: sum(a * b)
    // Dot Product: sum(a * b)
    EXPORT void prime_dot(long long n, double* res, double* a, double* b) {
        double sum = 0.0;
        // Correct syntax: separate parallel and simd, or combined 'parallel for simd'
        #pragma omp parallel for simd reduction(+:sum)
        for (long long i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        res[0] = sum;
    }

    // Magnitude: sqrt(sum(a^2))
    EXPORT void prime_mag(long long n, double* res, double* a) {
        double sum = 0.0;
        #pragma omp parallel for simd reduction(+:sum)
        for (long long i = 0; i < n; i++) {
            sum += a[i] * a[i];
        }
        res[0] = std::sqrt(sum);
    }

    // Normalize: a / magnitude(a)
    EXPORT void prime_normalize(long long n, double* res, double* a) {
        double mag = 0.0;
        #pragma omp parallel for reduction(+:mag)
        for (long long i = 0; i < n; i++) {
            mag += a[i] * a[i];
        }
        mag = std::sqrt(mag);

        // Avoid division by zero
        if (mag == 0.0) {
             #pragma omp parallel for
             for (long long i = 0; i < n; i++) res[i] = 0.0;
             return;
        }

        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = a[i] / mag;
        }
    }
}
