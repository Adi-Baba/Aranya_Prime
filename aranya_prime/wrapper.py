import ctypes
import os
import numpy as np
import platform

# Determine Library Name
system = platform.system()
if system == "Windows":
    lib_name = "aranya_prime.dll"
elif system == "Darwin":
    lib_name = "aranya_prime.dylib"
else:
    lib_name = "aranya_prime.so"

# Load Library
DLL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", lib_name)
try:
    _lib = ctypes.CDLL(DLL_PATH)
except OSError:
    raise RuntimeError(f"{lib_name} not found or failed to load. Run 'scripts/build.py'.")

# Helper to define argtypes
def _wrap(func_name, argtypes):
    try:
        func = getattr(_lib, func_name)
        func.argtypes = argtypes
        return func
    except AttributeError:
        return None

# Common types
c_n = ctypes.c_longlong
c_p = ctypes.POINTER(ctypes.c_double)

# --- 1. Polynomials ---
_prime_poly = _wrap("prime_poly", [c_n, c_p, c_p])

def polynomial(x):
    """Computes x^3 + x^2 + x"""
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    if _prime_poly:
        _prime_poly(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    else:
        raise NotImplementedError("Kernel 'prime_poly' not found.")
    return res

# --- 2. Trigonometry ---
_ops_trig = {
    "sin": _wrap("prime_sin", [c_n, c_p, c_p]),
    "cos": _wrap("prime_cos", [c_n, c_p, c_p]),
    "tan": _wrap("prime_tan", [c_n, c_p, c_p]),
}

def sin(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _ops_trig["sin"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

def cos(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _ops_trig["cos"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

def tan(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _ops_trig["tan"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

# --- 3. Array Ops ---
_ops_arr = {
    "add": _wrap("prime_math_sum", [c_n, c_p, c_p, c_p]), # Fortran
    "sub": _wrap("prime_sub", [c_n, c_p, c_p, c_p]),
    "mul": _wrap("prime_mul", [c_n, c_p, c_p, c_p]),
    "div": _wrap("prime_div", [c_n, c_p, c_p, c_p]),
}

def add(x, y): # Fortran
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    res = np.empty_like(x)
    if _ops_arr["add"]:
        # Fortran might crash if called with wrong ABI, but ISO_C_BINDING makes it standard C call
        _ops_arr["add"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    else:
        # Fallback to numpy or C++ if we ported it
        return x + y
    return res

def sub(x, y):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    res = np.empty_like(x)
    _ops_arr["sub"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

def mul(x, y):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    res = np.empty_like(x)
    _ops_arr["mul"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

def div(x, y):
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    res = np.empty_like(x)
    _ops_arr["div"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

# --- 4. Linear Algebra ---
_ops_linalg = {
    "dot": _wrap("prime_dot", [c_n, c_p, c_p, c_p]), # res is pointer to double (scalar)
    "mag": _wrap("prime_mag", [c_n, c_p, c_p]),      # res is pointer to double (scalar)
    "norm": _wrap("prime_normalize", [c_n, c_p, c_p]),
}

def dot(x, y):
    """Dot product (returns scalar)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    # Result is a single double
    res = np.zeros(1, dtype=np.float64)
    _ops_linalg["dot"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res[0]

def magnitude(x):
    """Euclidean norm / Magnitude."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.zeros(1, dtype=np.float64)
    _ops_linalg["mag"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res[0]

def normalize(x):
    """Returns normalized vector."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _ops_linalg["norm"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

# --- 5. Transforms ---
_ops_trans = {
    "scale": _wrap("prime_scale", [c_n, c_p, c_p, ctypes.c_double]),
    "rotate": _wrap("prime_rotate_2d", [c_n, c_p, c_p, c_p, c_p, ctypes.c_double]),
}

def scale(x, s):
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _ops_trans["scale"](x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), ctypes.c_double(s))
    return res

def rotate_2d(x, y, angle_rad):
    """
    Rotates points (x, y) by angle (radians).
    x: array of x coords
    y: array of y coords
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    res_x = np.empty_like(x)
    res_y = np.empty_like(y)
    
    _ops_trans["rotate"](x.size, 
                         res_x.ctypes.data_as(c_p), 
                         res_y.ctypes.data_as(c_p),
                         x.ctypes.data_as(c_p),
                         y.ctypes.data_as(c_p),
                         ctypes.c_double(angle_rad))
    return res_x, res_y
