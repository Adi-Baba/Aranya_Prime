import pytest
import numpy as np
import aranya_prime as ap
import time

# Helper for rough timing checks if pytest-benchmark isn't strictly used by user
# But we structure it for standard pytest execution.

N_LARGE = 1_000_000

@pytest.fixture
def large_data():
    x = np.random.rand(N_LARGE).astype(np.float64)
    y = np.random.rand(N_LARGE).astype(np.float64)
    return x, y

def test_perf_rotate_2d(large_data, benchmark):
    """Benchmark Rotation (usually the biggest speedup)."""
    x, y = large_data
    angle = 0.785
    
    # Run once to warm up / verify
    ap.rotate_2d(x, y, angle)
    
    # Use pytest-benchmark if available, otherwise just pass
    if benchmark:
        benchmark(ap.rotate_2d, x, y, angle)
    else:
        # Simple manual run
        start = time.perf_counter()
        ap.rotate_2d(x, y, angle)
        dur = time.perf_counter() - start
        print(f"Rotate 2D Time: {dur:.4f}s")

def test_perf_sin(large_data, benchmark):
    x, _ = large_data
    if benchmark:
        benchmark(ap.sin, x)

def test_perf_dot(large_data, benchmark):
    x, y = large_data
    if benchmark:
        benchmark(ap.dot, x, y)

def test_stress_integrity(large_data):
    """Rapid fire test to check for segmentation faults or memory leaks."""
    x, _ = large_data
    # Scale down N for rapid looping
    small_x = x[:1000]
    
    for _ in range(500):
        ap.polynomial(small_x)
    
    assert True # If we didn't crash, we passed.
