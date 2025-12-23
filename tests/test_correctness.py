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
    data = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.0], dtype=np.float64)
    # NumPy/C++ behavior: NaN->NaN, Inf->Inf
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
