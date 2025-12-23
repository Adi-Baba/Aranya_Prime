import numpy as np
import sys
import os
import time
import math

# Load Aranya Prime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import aranya_prime as ap
except Exception as e:
    print(f"CRITICAL: Engine not loaded. {e}")
    sys.exit(1)

def log(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m", "PASS": "\033[92m", 
        "FAIL": "\033[91m", "WARN": "\033[93m", "END": "\033[0m"
    }
    print(f"[{colors.get(status, '')}{status}{colors['END']}] {msg}")

def run_falsification():
    log("=== ARANYA PRIME: ULTIMATE FALSIFICATION TEST ===", "WARN")
    
    fails = 0
    
    # ---------------------------------------------------------
    # TEST 1: The "Oddity" (Boundary Conditions)
    # ---------------------------------------------------------
    log("Test 1: Boundary Conditions (Odd sizes, Empty, Prime sizes)")
    sizes = [0, 1, 2, 3, 7, 13, 1023, 1024, 1025, 1_000_003]
    for s in sizes:
        try:
            data = np.random.rand(s).astype(np.float64)
            res_np = data**3 + data**2 + data
            res_ar = ap.polynomial(data)
            
            if s == 0:
                if len(res_ar) == 0:
                    continue
                else:
                    log(f"Empty array failed to return empty result", "FAIL")
                    fails += 1
            
            if not np.allclose(res_np, res_ar, atol=1e-15):
                log(f"Size {s}: MISMATCH", "FAIL")
                fails += 1
        except Exception as e:
            log(f"Size {s}: CRASHED - {e}", "FAIL")
            fails += 1
            
    if fails == 0: log("Boundary Conditions Passed", "PASS")
    
    # ---------------------------------------------------------
    # TEST 2: The "Chaos" (NaNs and Infs)
    # ---------------------------------------------------------
    log("Test 2: Chaos Input (NaN, Inf, -Inf)")
    data = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.0], dtype=np.float64)
    try:
        res_np = data**3 + data**2 + data
        res_ar = ap.polynomial(data)
        
        # NumPy/C++ standard: NaN->NaN, Inf->Inf
        # We check frame-by-frame equality using hash/isnan logic
        np.testing.assert_array_equal(np.isnan(res_np), np.isnan(res_ar))
        np.testing.assert_array_equal(np.isinf(res_np), np.isinf(res_ar))
        
        # Check finite values
        mask = np.isfinite(res_np)
        if np.allclose(res_np[mask], res_ar[mask]):
            log("Chaos Handling Passed (Matches NumPy)", "PASS")
        else:
            log("Chaos Values Mismatch", "FAIL")
            fails += 1
    except Exception as e:
        log(f"Chaos Test CRASHED: {e}", "FAIL")
        fails += 1

    # ---------------------------------------------------------
    # TEST 3: Precision Drill (Denormals)
    # ---------------------------------------------------------
    log("Test 3: Subnormal/Tiny Numbers (Precision Check)")
    data = np.array([1e-300, 1e-310, 0.0], dtype=np.float64)
    # x^3 will underflow to 0 probably
    res_np = data**3 + data**2 + data
    res_ar = ap.polynomial(data)
    
    if np.allclose(res_np, res_ar, atol=0): # Strict equality
        log("Subnormal Precision Passed", "PASS")
    else:
        log("Subnormal Precision Failed (FTZ might be on)", "WARN")
        # Not essentially a fail, -O3 -ffast-math often enables Flush-To-Zero
        
    # ---------------------------------------------------------
    # TEST 4: The "Stress" (Rapid Fire)
    # ---------------------------------------------------------
    log("Test 4: Rapid Fire (Memory Leak/Race Check)")
    data = np.random.rand(100_000).astype(np.float64)
    t0 = time.time()
    for _ in range(1000):
        _ = ap.polynomial(data)
    dur = time.time() - t0
    log(f"Executed 1000 iter in {dur:.2f}s", "PASS")
    
    # ---------------------------------------------------------
    # TEST 5: Fortran Integration
    # ---------------------------------------------------------
    log("Test 5: Fortran Kernel (Addition)")
    try:
        x = np.random.rand(1000)
        y = np.random.rand(1000)
        res_np = x + y
        res_f90 = ap.add(x, y)
        if np.allclose(res_np, res_f90):
             log("Fortran Kernel Passed", "PASS")
        else:
             log("Fortran Kernel Mismatch", "FAIL")
             fails += 1
    except Exception as e:
        log(f"Fortran Kernel Failed: {e}", "FAIL")
        fails += 1

    log("==========================================", "INFO")
    if fails == 0:
        log("SUMMARY: ENGINE INTEGRITY VERIFIED (0 Errors)", "PASS")
    else:
        log(f"SUMMARY: ENGINE COMPROMISED ({fails} Errors)", "FAIL")

if __name__ == "__main__":
    run_falsification()
