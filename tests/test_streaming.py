import pytest
import numpy as np
import aranya_prime as ap
import time

def test_streaming_sin_correctness():
    N = 100_000
    x = np.random.rand(N).astype(np.float64)
    
    res_std = ap.sin(x)
    res_str = ap.stream.sin(x, chunk_size=1024) # tiny chunks to force many transitions
    
    np.testing.assert_allclose(res_str, res_std, atol=1e-15)

def test_streaming_rotate_correctness():
    N = 100_000
    x = np.random.rand(N).astype(np.float64)
    y = np.random.rand(N).astype(np.float64)
    angle = 0.5
    
    rx1, ry1 = ap.rotate_2d(x, y, angle)
    rx2, ry2 = ap.stream.rotate_2d(x, y, angle, chunk_size=2048)
    
    np.testing.assert_allclose(rx1, rx2, atol=1e-15)
    np.testing.assert_allclose(ry1, ry2, atol=1e-15)

def test_giant_array_comp():
    """Test with 10M elements (~80MB), likely exceeding L3 cache."""
    N = 10_000_000
    x = np.random.rand(N).astype(np.float64)
    
    start = time.perf_counter()
    ap.sin(x)
    t_std = time.perf_counter() - start
    
    start = time.perf_counter()
    ap.stream.sin(x, chunk_size=65536) # ~512KB chunks, fits in L2/L3
    t_str = time.perf_counter() - start
    
    print(f"\n[Giant Array 10M] Standard: {t_std:.4f}s, Streaming: {t_str:.4f}s")
    # Note: On some architectures, standard Rayon might already be cache-aware 
    # due to how it splits work, but explicit chunking offers a floor for performance.
