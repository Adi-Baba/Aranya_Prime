import pytest
import numpy as np
import aranya_prime as ap
import math

# --- 1. Boundary Conditions ---
@pytest.mark.parametrize("size", [0, 1, 2, 3, 7, 1024, 1_000_003])
def test_boundary_conditions(size):
    """Test odd sizes, empty arrays, and prime sizes."""
    data = np.random.rand(size).astype(np.float64)
    res_np = data**3 + data**2 + data
    res_ar = ap.polynomial(data)
    
    if size == 0:
        assert len(res_ar) == 0
    else:
        np.testing.assert_allclose(res_ar, res_np, atol=1e-15)

# --- 2. Chaos (NaNs and Infs) ---
def test_chaos_handling():
    """Ensure NaNs and Infs are propagated correctly (IEEE-754)."""
    import warnings
    data = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.0], dtype=np.float64)

    # NumPy raises a RuntimeWarning for NaN arithmetic — suppress it since
    # we already expect NaN in the output. This is NumPy's own warning, not Aranya's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res_np = data**3 + data**2 + data

    res_ar = ap.polynomial(data)

    # Check structure identical (NaN locations match)
    np.testing.assert_array_equal(np.isnan(res_ar), np.isnan(res_np))
    np.testing.assert_array_equal(np.isinf(res_ar), np.isinf(res_np))
    
    # Check finite values match
    mask = np.isfinite(res_np)
    np.testing.assert_allclose(res_ar[mask], res_np[mask], atol=1e-15)

# --- 3. Precision (Denormals) ---
def test_subnormal_precision():
    """Check handling of tiny numbers (close to 0)."""
    data = np.array([1e-300, 1e-310, 0.0], dtype=np.float64)
    res_np = data**3 + data**2 + data
    res_ar = ap.polynomial(data)
    # Strict equality expected for these small values
    np.testing.assert_allclose(res_ar, res_np, atol=0)

# --- 4. Core Functions ---
def test_trigonometry():
    N = 1000
    x = np.random.rand(N)
    
    np.testing.assert_allclose(ap.sin(x), np.sin(x), atol=1e-15)
    np.testing.assert_allclose(ap.cos(x), np.cos(x), atol=1e-15)
    np.testing.assert_allclose(ap.tan(x), np.tan(x), atol=1e-15)

def test_linear_algebra():
    N = 1000
    a = np.random.rand(N)
    b = np.random.rand(N)
    
    # Dot Product
    res_dot = ap.dot(a, b)
    ref_dot = np.dot(a, b)
    # Summation error grows with N, relative error is better check
    if abs(ref_dot) > 1e-9:
        assert abs((res_dot - ref_dot) / ref_dot) < 1e-12
    else:
        assert abs(res_dot - ref_dot) < 1e-12

    # Magnitude
    np.testing.assert_allclose(ap.magnitude(a), np.linalg.norm(a), atol=1e-12)

def test_transforms():
    N = 1000
    x = np.random.rand(N)
    y = np.random.rand(N)
    angle = 0.5
    
    # Rotate
    rx, ry = ap.rotate_2d(x, y, angle)
    
    c, s = np.cos(angle), np.sin(angle)
    ref_x = x * c - y * s
    ref_y = x * s + y * c
    
    np.testing.assert_allclose(rx, ref_x, atol=1e-15)
    np.testing.assert_allclose(ry, ref_y, atol=1e-15)

# --- 5. Fortran Integration ---
def test_fortran_ops():
    N = 1000
    a = np.random.rand(N)
    b = np.random.rand(N)
    
    np.testing.assert_allclose(ap.add(a, b), np.add(a, b), atol=1e-15)
    np.testing.assert_allclose(ap.sub(a, b), np.subtract(a, b), atol=1e-15)
    np.testing.assert_allclose(ap.mul(a, b), np.multiply(a, b), atol=1e-15)
    
    # Div (handle zero/small check implicitly via random, generic logic)
    b_safe = b + 1.0 # avoid div by zero for test
    np.testing.assert_allclose(ap.div(a, b_safe), np.divide(a, b_safe), atol=1e-15)

def test_safety_mismatch():
    """Ensure disparate array sizes raise ValueError instead of segfault."""
    a = np.zeros(10)
    b = np.zeros(11)
    
    with pytest.raises(ValueError, match="Array size mismatch"):
        ap.add(a, b)
    
    with pytest.raises(ValueError):
        ap.dot(a, b)

# --- DCT & Wavelet ---
def test_dct_matches_numpy_formula():
    """Verify DCT against a reference NumPy-based formula (orthonormal DCT-II)."""
    x = np.random.rand(128).astype(np.float64)
    n = x.shape[0]

    # Reference DCT-II (orthonormal):
    #   X[k] = alpha[k] * sum_{i=0..n-1} x[i] * cos(pi*(i+0.5)*k / n)
    # where alpha[0]=sqrt(1/n), alpha[k>0]=sqrt(2/n).
    k = np.arange(n)
    i = np.arange(n)
    cos_matrix = np.cos(np.pi * (i[:, None] + 0.5) * k[None, :] / n)
    alpha = np.sqrt(2.0 / n) * np.ones(n)
    alpha[0] = np.sqrt(1.0 / n)
    ref = alpha * (cos_matrix.T @ x)

    res = ap.dct(x)
    np.testing.assert_allclose(res, ref, atol=1e-10)

def test_wavelet_haar_single_level():
    # Haar transform: single-level, even length
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # Manual calculation:
    # Approx: [(1+2)/sqrt(2), (3+4)/sqrt(2)]
    # Detail: [(1-2)/sqrt(2), (3-4)/sqrt(2)]
    sqrt2 = np.sqrt(2)
    ref = np.array([(1+2)/sqrt2, (3+4)/sqrt2, (1-2)/sqrt2, (3-4)/sqrt2])
    res = ap.wavelet_transform(x)
    np.testing.assert_allclose(res, ref, atol=1e-12)
    # Test error on odd length
    with pytest.raises(ValueError):
        ap.wavelet_transform(np.arange(5, dtype=np.float64))

