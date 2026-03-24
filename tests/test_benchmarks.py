import numpy as np
import pytest
import aranya_prime as ap

from typing import Any

try:
    import numba
    from numba import njit
except ImportError:  # pragma: no cover
    numba = None
    njit: Any = None

# Benchmark comparisons between NumPy and Aranya Prime kernels.
# Use `--benchmark-group-by=group` to see side-by-side comparisons.

@pytest.fixture(scope="module")
def rng():
    rng = np.random.default_rng(1234)
    return rng


@pytest.mark.benchmark(group="dot")
def test_dot_numpy(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    y = rng.random(1_000_000, dtype=np.float64)
    benchmark(np.dot, x, y)


@pytest.mark.benchmark(group="dot")
def test_dot_aranya(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    y = rng.random(1_000_000, dtype=np.float64)
    benchmark(ap.dot, x, y, False)


@pytest.mark.benchmark(group="dot")
def test_dot_aranya_blas(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    y = rng.random(1_000_000, dtype=np.float64)
    benchmark(ap.dot, x, y, True)


@pytest.mark.benchmark(group="dot")
@pytest.mark.skipif(numba is None, reason="numba not installed")
def test_dot_numba(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    y = rng.random(1_000_000, dtype=np.float64)

    @njit
    def dot(a, b):
        s = 0.0
        for i in range(a.shape[0]):
            s += a[i] * b[i]
        return s

    # Warm-up compile
    dot(x, y)
    benchmark(dot, x, y)


@pytest.mark.benchmark(group="matmul")
def test_matmul_numpy(benchmark, rng):
    A = rng.random((512, 512), dtype=np.float64)
    B = rng.random((512, 512), dtype=np.float64)
    benchmark(np.dot, A, B)


@pytest.mark.benchmark(group="matmul")
def test_matmul_aranya_rayon(benchmark, rng):
    A = rng.random((512, 512), dtype=np.float64)
    B = rng.random((512, 512), dtype=np.float64)
    benchmark(ap.matmul, A, B, False)


@pytest.mark.benchmark(group="matmul")
def test_matmul_aranya_blas(benchmark, rng):
    A = rng.random((512, 512), dtype=np.float64)
    B = rng.random((512, 512), dtype=np.float64)
    benchmark(ap.matmul, A, B, True)


@pytest.mark.benchmark(group="matmul")
@pytest.mark.skipif(numba is None, reason="numba not installed")
def test_matmul_numba(benchmark, rng):
    A = rng.random((512, 512), dtype=np.float64)
    B = rng.random((512, 512), dtype=np.float64)

    @njit
    def matmul(a, b):
        m, k = a.shape
        k2, n = b.shape
        out = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += a[i, p] * b[p, j]
                out[i, j] = s
        return out

    # Warm-up compile
    _ = matmul(A, B)
    benchmark(matmul, A, B)


@pytest.mark.benchmark(group="fft")
def test_fft_numpy(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    benchmark(np.fft.fft, x)


@pytest.mark.benchmark(group="fft")
def test_fft_aranya(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    benchmark(ap.fft, x)


@pytest.mark.benchmark(group="fft")
@pytest.mark.skipif(numba is None, reason="numba not installed")
def test_fft_numba(benchmark, rng):
    # Numba FFT is O(n^2) in this naive implementation, so we keep it small.
    n = 4096
    real = rng.random(n, dtype=np.float64)
    imag = rng.random(n, dtype=np.float64)
    x = real + 1j * imag

    @njit
    def fft_simple(z):
        n = z.shape[0]
        out = np.zeros(n, dtype=np.complex128)
        for k in range(n):
            s = 0 + 0j
            for t in range(n):
                angle = -2j * np.pi * k * t / n
                s += z[t] * np.exp(angle)
            out[k] = s
        return out

    # Warm-up compile
    fft_simple(np.ones(16, dtype=np.complex128))
    benchmark(fft_simple, x)


@pytest.mark.benchmark(group="dct")
def test_dct_numpy(benchmark, rng):
    # Use an FFT-based reference implementation (avoids SciPy dependency).
    x = rng.random(1_000_000, dtype=np.float64)
    n = x.shape[0]

    def dct_ref(x):
        n = x.shape[0]
        y = np.empty(2 * n, dtype=np.float64)
        y[:n] = x
        y[n:] = x[::-1]
        Y = np.fft.fft(y)
        k = np.arange(n)
        factor = np.exp(-1j * np.pi * k / (2 * n))
        alpha = np.sqrt(2.0 / n) * np.ones(n, dtype=np.float64)
        alpha[0] = np.sqrt(1.0 / n)
        return (alpha * (factor * Y[:n]).real)

    benchmark(dct_ref, x)


@pytest.mark.benchmark(group="dct")
def test_dct_aranya(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    benchmark(ap.dct, x)


@pytest.mark.benchmark(group="wavelet")
def test_wavelet_python(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    if x.shape[0] % 2:
        x = x[:-1]

    def wavelet_ref(x):
        n = x.shape[0]
        sqrt2 = np.sqrt(2.0)
        out = np.empty(n, dtype=np.float64)
        out[: n // 2] = (x[0::2] + x[1::2]) / sqrt2
        out[n // 2 :] = (x[0::2] - x[1::2]) / sqrt2
        return out

    benchmark(wavelet_ref, x)


@pytest.mark.benchmark(group="wavelet")
def test_wavelet_aranya(benchmark, rng):
    x = rng.random(1_000_000, dtype=np.float64)
    if x.shape[0] % 2:
        x = x[:-1]
    benchmark(ap.wavelet_transform, x)


@pytest.mark.benchmark(group="svd")
def test_svd_numpy(benchmark, rng):
    A = rng.random((256, 256), dtype=np.float64)
    benchmark(np.linalg.svd, A, full_matrices=False)


@pytest.mark.benchmark(group="svd")
def test_svd_aranya(benchmark, rng):
    A = rng.random((256, 256), dtype=np.float64)
    benchmark(ap.svd, A)


def test_blas_backend_info():
    info = ap.blas_info()
    print("BLAS backend info:", info)
    assert "backend" in info
    assert "config_output" in info


@pytest.mark.benchmark(group="dot-scaling")
@pytest.mark.parametrize("n", [100_000, 500_000, 1_000_000])
def test_dot_scaling_numpy(benchmark, rng, n):
    x = rng.random(n, dtype=np.float64)
    y = rng.random(n, dtype=np.float64)
    benchmark(np.dot, x, y)


@pytest.mark.benchmark(group="dot-scaling")
@pytest.mark.parametrize("n", [100_000, 500_000, 1_000_000])
def test_dot_scaling_aranya(benchmark, rng, n):
    x = rng.random(n, dtype=np.float64)
    y = rng.random(n, dtype=np.float64)
    benchmark(ap.dot, x, y, True)


@pytest.mark.benchmark(group="matmul-scaling")
@pytest.mark.parametrize("dim", [128, 256])
def test_matmul_scaling_numpy(benchmark, rng, dim):
    A = rng.random((dim, dim), dtype=np.float64)
    B = rng.random((dim, dim), dtype=np.float64)
    benchmark(np.dot, A, B)


@pytest.mark.benchmark(group="matmul-scaling")
@pytest.mark.parametrize("dim", [128, 256])
def test_matmul_scaling_aranya(benchmark, rng, dim):
    A = rng.random((dim, dim), dtype=np.float64)
    B = rng.random((dim, dim), dtype=np.float64)
    benchmark(ap.matmul, A, B, True)
