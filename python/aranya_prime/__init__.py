"""
Aranya Prime (Rust Edition)
High-performance computational kernels powered by Rust, PyO3, and Rayon.
"""

from ._aranya_prime import (
    prime_poly,
    prime_sin, prime_cos, prime_tan,
    prime_math_sum, prime_sub, prime_mul, prime_div,
    prime_dot, prime_mag, prime_normalize,
    prime_matmul, prime_normalize_batch,
    prime_scale, prime_rotate_2d,
    prime_sum, prime_mean, prime_std, prime_clip,
    prime_l2_norm, prime_linf_norm,
    prime_convolve,
    prime_fft, prime_ifft,
    prime_dct, prime_wavelet_transform,
    # f32 variants
    prime_sin_f32, prime_cos_f32, prime_tan_f32,
    prime_dot_f32, prime_matmul_f32, prime_rotate_2d_f32,
    # streaming
    prime_chunked_sin, prime_chunked_rotate_2d,
    # BLAS / LAPACK
    prime_blas_dot, prime_blas_matmul, prime_svd,
)

# ── Polynomials ────────────────────────────────────────────────────────────────
def polynomial(x):
    """Evaluates x³ + x² + x element-wise (Rayon parallel)."""
    return prime_poly(x)

# ── Trigonometry ──────────────────────────────────────────────────────────────
def sin(x): return prime_sin(x)
def cos(x): return prime_cos(x)
def tan(x): return prime_tan(x)

# ── Element-wise Array Operations ─────────────────────────────────────────────
def add(x, y): return prime_math_sum(x, y)
def sub(x, y): return prime_sub(x, y)
def mul(x, y): return prime_mul(x, y)
def div(x, y): return prime_div(x, y)

# ── Statistics & Norms ────────────────────────────────────────────────────────
def sum(x):   return prime_sum(x)
def mean(x):  return prime_mean(x)
def std(x):   return prime_std(x)
def clip(x, min_val, max_val): return prime_clip(x, min_val, max_val)
def l2_norm(x):   return prime_l2_norm(x)
def linf_norm(x): return prime_linf_norm(x)

# ── Linear Algebra (1D) ───────────────────────────────────────────────────────
def dot(x, y, auto_blas=True):
    """
    Dot product. 
    Auto-dispatches to BLAS if size > 1M elements.
    """
    if auto_blas and len(x) > 1_000_000:
        return prime_blas_dot(x, y)
    return prime_dot(x, y)

def magnitude(x):
    """Euclidean norm (L2) of a 1D vector."""
    return prime_mag(x)

def normalize(x):
    """Returns a unit-length version of x."""
    return prime_normalize(x)

# ── Linear Algebra (2D) ───────────────────────────────────────────────────────
def matmul(A, B, auto_dispatch=True):
    """
    Matrix multiplication C = A @ B.
    By default, dispatches large matrices to Fortran BLAS (dgemm).
    Small matrices use the Rayon thread-pool for lower latency.
    """
    if auto_dispatch:
        m, k = A.shape
        k2, n = B.shape
        # Heuristic: BLAS is worth it for > 500k total mult-adds
        if m * n * k > 500_000:
            return prime_blas_matmul(A, B)
    return prime_matmul(A, B)

def svd(A):
    """
    Fortran-backed SVD via LAPACK (dgesvd).
    Returns (U, S, Vh) matching np.linalg.svd.
    """
    return prime_svd(A)

def normalize_batch(X):
    """Row-wise L2 normalization of a 2D matrix."""
    return prime_normalize_batch(X)

# ── Transforms ────────────────────────────────────────────────────────────────
def scale(x, s):
    """Scales every element by scalar s."""
    return prime_scale(x, s)

def rotate_2d(x, y, angle_rad):
    """Rotates 2D points (x, y) by angle_rad. Returns (new_x, new_y)."""
    return prime_rotate_2d(x, y, angle_rad)

# ── Signal Processing ─────────────────────────────────────────────────────────
# ── Signal Processing ─────────────────────────────────────────────────────────
def convolve(signal, kernel):
    """1D direct convolution (full mode). Fast for small kernels."""
    return prime_convolve(signal, kernel)


def fft(x):
    """Forward FFT. Returns (real_part, imag_part) tuple."""
    return prime_fft(x)

def ifft(re, im):
    """Inverse FFT from (real, imag). Returns reconstructed real signal."""
    return prime_ifft(re, im)

def dct(x):
    """Discrete Cosine Transform (DCT-II). Returns DCT coefficients."""
    return prime_dct(x)

def wavelet_transform(x):
    """Haar wavelet transform (single-level). Returns [approx..., detail...]."""
    return prime_wavelet_transform(x)

# ── f32 Fast-Math Sub-namespace ──────────────────────────────────────────────
class F32Namespace:
    """Sub-namespace for single-precision (f32) kernels."""
    @staticmethod
    def sin(x): return prime_sin_f32(x)
    @staticmethod
    def cos(x): return prime_cos_f32(x)
    @staticmethod
    def tan(x): return prime_tan_f32(x)
    @staticmethod
    def dot(x, y): return prime_dot_f32(x, y)
    @staticmethod
    def matmul(A, B): return prime_matmul_f32(A, B)
    @staticmethod
    def rotate_2d(x, y, angle): return prime_rotate_2d_f32(x, y, angle)

f32 = F32Namespace()

# ── Streaming / Chunked Sub-namespace ──────────────────────────────────────────
class StreamNamespace:
    """Sub-namespace for streaming/chunked kernels (large array optimized)."""
    @staticmethod
    def sin(x, chunk_size=65536): 
        """Parallel sin with explicit cache-sized chunks."""
        return prime_chunked_sin(x, chunk_size)
    @staticmethod
    def rotate_2d(x, y, angle_rad, chunk_size=65536):
        """Parallel 2D rotation with explicit cache-sized chunks."""
        return prime_chunked_rotate_2d(x, y, angle_rad, chunk_size)

def blas_info():
    """Return information about the BLAS/LAPACK backend used by NumPy.

    This is useful for understanding what BLAS implementation is being used
    (OpenBLAS / MKL / etc.) and where it is loaded from.
    """
    import io
    import sys
    import numpy as _np

    # Capture numpy configuration output, which includes BLAS/LAPACK info.
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        _np.show_config()
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue().strip()

    backend = "unknown"
    if "openblas" in output.lower():
        backend = "OpenBLAS"
    elif "mkl" in output.lower():
        backend = "MKL"
    elif "atlas" in output.lower():
        backend = "ATLAS"

    return {
        "numpy_version": _np.__version__,
        "backend": backend,
        "config_output": output,
    }


stream = StreamNamespace()
