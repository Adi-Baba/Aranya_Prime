import numpy as np
import time
import sys
import os

# Load Aranya Prime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import aranya_prime as ap
    print("[INIT] Aranya Prime Loaded Successfully.")
except Exception as e:
    print(f"[FAIL] Engine not loaded. {e}")
    sys.exit(1)

def benchmark(name, func, *args, ref_func=None):
    """
    Runs func(*args), checks against ref_func(*args) if provided, and prints time.
    """
    print(f"\n--- {name} ---")
    
    # Run Prime
    start = time.perf_counter()
    res_prime = func(*args)
    duration = time.perf_counter() - start
    
    print(f"Time: {duration:.4f}s")
    
    # Validation
    if ref_func:
        # NumPy Baseline
        start_np = time.perf_counter()
        if isinstance(args[0], tuple): # Handle multi-return unpack
             res_ref = ref_func(*args)
        else:
             res_ref = ref_func(*args)
        np_duration = time.perf_counter() - start_np
        
        speedup = np_duration / duration
        print(f"Speedup vs NumPy: {speedup:.2f}x")
        
        # Check correctness (Relative Error preferred for large arrays/values)
        if isinstance(res_prime, tuple):
            # for rotate_2d which returns (x, y)
            prim_x, prim_y = res_prime
            ref_x, ref_y = res_ref
            # Check X
            diff_x = np.abs(prim_x - ref_x)
            max_err_x = np.max(diff_x)
            # Check Y
            diff_y = np.abs(prim_y - ref_y)
            max_err_y = np.max(diff_y)
            max_err = max(max_err_x, max_err_y)
        elif np.isscalar(res_prime) or res_prime.shape == (1,):
            # Scalar result (dot product)
            diff = np.abs(res_prime - res_ref)
            # Use relative error for large sums
            if np.abs(res_ref) > 1e-9:
                rel_err = diff / np.abs(res_ref)
                max_err = rel_err
            else:
                max_err = diff
        else:
            # Array Result
            # Use relative error for safety if needed, but for stress test abs is quick indication
            diff = np.abs(res_prime - res_ref)
            max_err = np.max(diff)

        # Allow small deviation (Float precision)
        if max_err < 1e-9:
            print(f"\033[92m[PASS]\033[0m Max Error: {max_err:.2e}")
        else:
            print(f"\033[91m[FAIL]\033[0m Max Error: {max_err:.2e}")

def main():
    N = 10_000_000
    print(f"Generatign Data (N={N:,})...")
    
    # Random Data - large enough to stress memory bandwidth
    a = np.random.rand(N).astype(np.float64)
    b = np.random.rand(N).astype(np.float64)
    angle = 0.785398 # 45 degrees
    
    # 1. Trigonometry
    benchmark("Sin", ap.sin, a, ref_func=np.sin)
    benchmark("Cos", ap.cos, a, ref_func=np.cos)
    benchmark("Tan", ap.tan, a, ref_func=np.tan)
    
    # 2. Array Ops
    benchmark("Add (Fortran)", ap.add, a, b, ref_func=np.add)
    benchmark("Sub", ap.sub, a, b, ref_func=np.subtract)
    benchmark("Mul", ap.mul, a, b, ref_func=np.multiply)
    benchmark("Div", ap.div, a, b, ref_func=np.divide)
    
    # 3. Linear Algebra
    benchmark("Dot Product", ap.dot, a, b, ref_func=np.dot)
    benchmark("Magnitude", ap.magnitude, a, ref_func=np.linalg.norm)
    
    def normalize_ref(v):
        norm = np.linalg.norm(v)
        if norm == 0: return v
        return v / norm
        
    benchmark("Normalize", ap.normalize, a, ref_func=normalize_ref)
    
    # 4. Transformations
    def scale_ref(v, s): return v * s
    benchmark("Scale", ap.scale, a, 2.5, ref_func=scale_ref)
    
    def rotate_ref(x, y, rad):
        c, s = np.cos(rad), np.sin(rad)
        rx = x * c - y * s
        ry = x * s + y * c
        return rx, ry

    benchmark("Rotate 2D", ap.rotate_2d, a, b, angle, ref_func=rotate_ref)
    
    print("\n[DONE] All tests completed.")

if __name__ == "__main__":
    main()
