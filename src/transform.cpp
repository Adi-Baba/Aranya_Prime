#include <cmath>
#include <omp.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

extern "C" {
    
    // Scale: res = a * scalar
    EXPORT void prime_scale(long long n, double* res, double* a, double scalar) {
        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            res[i] = a[i] * scalar;
        }
    }

    // Rotate 2D: Rotates points (x, y) by angle (radians)
    // res_x = x * cos(theta) - y * sin(theta)
    // res_y = x * sin(theta) + y * cos(theta)
    EXPORT void prime_rotate_2d(long long n, double* res_x, double* res_y, double* in_x, double* in_y, double angle) {
        double c = std::cos(angle);
        double s = std::sin(angle);

        #pragma omp parallel for
        for (long long i = 0; i < n; i++) {
            double x = in_x[i];
            double y = in_y[i];
            res_x[i] = x * c - y * s;
            res_y[i] = x * s + y * c;
        }
    }
}
