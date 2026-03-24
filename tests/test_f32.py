import pytest
import numpy as np
import aranya_prime as ap

def test_f32_sin():
    x = np.random.rand(1000).astype(np.float32)
    res = ap.f32.sin(x)
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, np.sin(x), atol=1e-6)

def test_f32_dot():
    a = np.random.rand(1000).astype(np.float32)
    b = np.random.rand(1000).astype(np.float32)
    res = ap.f32.dot(a, b)
    ref = np.dot(a, b)
    assert isinstance(res, float)
    assert abs(res - ref) < 1e-3 # float32 summation error can be high

def test_f32_matmul():
    A = np.random.rand(64, 128).astype(np.float32)
    B = np.random.rand(128, 32).astype(np.float32)
    res = ap.f32.matmul(A, B)
    ref = np.matmul(A, B)
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, ref, atol=1e-5)

def test_f32_rotate():
    x = np.random.rand(1000).astype(np.float32)
    y = np.random.rand(1000).astype(np.float32)
    angle = 0.5
    rx, ry = ap.f32.rotate_2d(x, y, angle)
    
    c, s = np.cos(angle), np.sin(angle)
    ref_x = x * c - y * s
    ref_y = x * s + y * c
    
    assert rx.dtype == np.float32
    assert ry.dtype == np.float32
    np.testing.assert_allclose(rx, ref_x, atol=1e-6)
    np.testing.assert_allclose(ry, ref_y, atol=1e-6)

# Performance comparison (informal)
def test_perf_comparison(benchmark):
    N = 1_000_000
    x_f64 = np.random.rand(N).astype(np.float64)
    x_f32 = x_f64.astype(np.float32)
    
    @benchmark
    def run_f32():
        ap.f32.sin(x_f32)
    
    # Just for recording, we don't assert speed here, 
    # but f32 should generally be faster due to cache density and SIMD.
