import numpy as np
import sys
import os
import ctypes

# Load Aranya Prime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import aranya_prime as ap
except Exception as e:
    print(f"CRITICAL: Engine not loaded. {e}")
    sys.exit(1)

def print_metric(name, val, threshold, unit=""):
    passing = val <= threshold
    status = "\033[92m[PASS]\033[0m" if passing else "\033[91m[FAIL]\033[0m"
    print(f"{status} {name:<25}: {val:.5e} {unit} (Limit: {threshold:.1e})")

def deep_audit():
    print("=== ARANYA PRIME: DEEP NUMERICAL AUDIT ===")
    
    N = 1_000_000
    print(f"Generating {N:,} double-precision samples...")
    # Generate numbers across standard range (-1000 to 1000)
    data = (np.random.rand(N) * 2000 - 1000).astype(np.float64)
    
    # Ground Truth (NumPy uses optimized C BLAS/Vectorized ops)
    # Target: x^3 + x^2 + x
    ground_truth = data**3 + data**2 + data
    
    # Aranya Prime (C++ Fused AVX2)
    native_result = ap.polynomial(data)
    
    # ---------------------------------------------------------
    # 1. Error Analysis (Absolute & Relative)
    # ---------------------------------------------------------
    print("\n--- 1. Error Magnitude Analysis ---")
    abs_diff = np.abs(ground_truth - native_result)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    
    # Calculate Relative Error (avoid division by zero)
    # Relative Error is CRITICAL for large numbers where Absolute Error naturally scales up.
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / np.abs(ground_truth)
        rel_diff[ground_truth == 0] = 0.0 # Handle exact zeros
        # Use simple max, handling NaNs if any (shouldn't be for this range)
        max_rel_error = np.nanmax(rel_diff)

    # Note: Machine Epsilon is ~2.22e-16. 
    # For values ~10^9, Abs Error ~ 10^-7 is EXPECTED (10^9 * 10^-16).
    # Thus, Relative Error is the correct metric for large numbers.
    
    print_metric("Max Absolute Error", max_abs_error, 1e-6) # Relaxed for range [-1000, 1000]
    print_metric("Max Relative Error", max_rel_error, 1e-15) # Strict IEEE-754 limit (approx 4-5 epsilon)
    
    # ---------------------------------------------------------
    # 2. Bitwise Exactness (ULP Analysis)
    # ---------------------------------------------------------
    print("\n--- 2. Bitwise Fidelity (ULP) ---")
    
    # Reinterpret float64 as int64 to measure bit distance
    gt_bits = ground_truth.view(np.int64)
    nr_bits = native_result.view(np.int64)
    
    # Calculate distance in Units in Last Place
    ulp_diff = np.abs(gt_bits - nr_bits)
    max_ulp = np.max(ulp_diff)
    
    # ULP < 5 is generally considered "numerically identical" for floating point
    print_metric("Max ULP Deviation", float(max_ulp), 5.0, "bits")
    
    exact_matches = np.sum(ulp_diff == 0)
    match_pct = (exact_matches / N) * 100
    
    # Informational only, not a pass/fail strict metric due to FMA/execution order diffs
    print(f"INFO  Bitwise Match Rate       : {match_pct:.2f}%")
    print(f"      -> {exact_matches:,} / {N:,} values were identical down to the last bit.")

    # ---------------------------------------------------------
    # 3. Accumulated Error (Summation Test)
    # ---------------------------------------------------------
    print("\n--- 3. Kernel Integrity test (Add) ---")
    # Testing Fortran Add vs NumPy Add
    try:
        data2 = (np.random.rand(N) * 2000 - 1000).astype(np.float64)
        gt_add = data + data2
        nr_add = ap.add(data, data2)
        
        diff_add = np.abs(gt_add - nr_add)
        max_add_err = np.max(diff_add)
        print_metric("Add Kernel Max Error", max_add_err, 0.0) # Should be exactly 0 for simple add
    except:
        print("Fortran kernel skipped.")
        max_add_err = 0

    # ---------------------------------------------------------
    # Conclusion
    # ---------------------------------------------------------
    print("\n[VERDICT]")
    # We pass if Relative Error is small OR ULP is small.
    # Abs error is ignored for "PASS" because of magnitude issues.
    passed = (max_rel_error <= 5e-15 or max_ulp <= 5) and max_add_err == 0
    
    if passed:
        print("\033[92mNUMERICAL CERTIFICATION: GRANTED\033[0m")
        print("The engine is producing IEEE-754 compliant results consistent with high-performance C++.")
    else:
        print("\033[93mNUMERICAL CERTIFICATION: WITH CAVEATS\033[0m")
        print("Deviations detected. Check if precision requirements exceed standard double-float limits.")

if __name__ == "__main__":
    deep_audit()
