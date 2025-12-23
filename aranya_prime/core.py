import ctypes
import os
import numpy as np
import numpy.typing as npt

import platform

# Determine Library Name
system = platform.system()
if system == "Windows":
    lib_name = "aranya_prime.dll"
elif system == "Darwin":
    lib_name = "aranya_prime.dylib"
else:
    lib_name = "aranya_prime.so"

# Lazy library loading - allows import before build
_lib = None
_lib_loaded = False

def _get_lib():
    """Load the native library on first use."""
    global _lib, _lib_loaded
    if _lib_loaded:
        return _lib
    
    # Try multiple paths: installed package, dev source, bin folder
    possible_paths = [
        os.path.join(os.path.dirname(__file__), lib_name),  # Inside package
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", lib_name),  # Dev: ../bin/
        os.path.join(os.path.dirname(__file__), "bin", lib_name),  # Package bin subfolder
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                _lib = ctypes.CDLL(path)
                _lib_loaded = True
                return _lib
            except OSError:
                continue
    
    raise RuntimeError(
        f"{lib_name} not found. Run 'python scripts/build.py' to compile native kernels.\n"
        f"Searched paths:\n" + "\n".join(f"  - {p}" for p in possible_paths)
    )

# Helper to define argtypes (lazy)
def _wrap(func_name, argtypes):
    def wrapper(*args):
        lib = _get_lib()
        try:
            func = getattr(lib, func_name)
            func.argtypes = argtypes
            return func(*args)
        except AttributeError:
            raise NotImplementedError(f"Kernel '{func_name}' not found in native library.")
    return wrapper

# Common types
c_n = ctypes.c_longlong
c_p = ctypes.POINTER(ctypes.c_double)
NDArrayFloat = npt.NDArray[np.float64]

def _check_equal_size(x: NDArrayFloat, y: NDArrayFloat):
    if x.size != y.size:
        raise ValueError(f"Array size mismatch: {x.size} != {y.size}")

# --- 1. Polynomials ---
def polynomial(x: NDArrayFloat) -> NDArrayFloat:

    """Computes x^3 + x^2 + x"""
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _wrap("prime_poly", [c_n, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

# --- 2. Trigonometry ---
def sin(x: NDArrayFloat) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _wrap("prime_sin", [c_n, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

def cos(x: NDArrayFloat) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _wrap("prime_cos", [c_n, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

def tan(x: NDArrayFloat) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _wrap("prime_tan", [c_n, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

# --- 3. Array Ops ---
def add(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:

    """Element-wise addition (Fortran kernel)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _check_equal_size(x, y)
    res = np.empty_like(x)
    _wrap("prime_math_sum", [c_n, c_p, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

def sub(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _check_equal_size(x, y)
    res = np.empty_like(x)
    _wrap("prime_sub", [c_n, c_p, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

def mul(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _check_equal_size(x, y)
    res = np.empty_like(x)
    _wrap("prime_mul", [c_n, c_p, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

def div(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _check_equal_size(x, y)
    res = np.empty_like(x)
    _wrap("prime_div", [c_n, c_p, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res

# --- 4. Linear Algebra ---
def dot(x: NDArrayFloat, y: NDArrayFloat) -> float:

    """Dot product (returns scalar)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _check_equal_size(x, y)
    res = np.zeros(1, dtype=np.float64)
    _wrap("prime_dot", [c_n, c_p, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), y.ctypes.data_as(c_p))
    return res[0]

def magnitude(x: NDArrayFloat) -> float:

    """Euclidean norm / Magnitude."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.zeros(1, dtype=np.float64)
    _wrap("prime_mag", [c_n, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res[0]

def normalize(x: NDArrayFloat) -> NDArrayFloat:

    """Returns normalized vector."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _wrap("prime_normalize", [c_n, c_p, c_p])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p))
    return res

# --- 5. Transforms ---
def scale(x: NDArrayFloat, s: float) -> NDArrayFloat:

    x = np.ascontiguousarray(x, dtype=np.float64)
    res = np.empty_like(x)
    _wrap("prime_scale", [c_n, c_p, c_p, ctypes.c_double])(x.size, res.ctypes.data_as(c_p), x.ctypes.data_as(c_p), ctypes.c_double(s))
    return res

def rotate_2d(x: NDArrayFloat, y: NDArrayFloat, angle_rad: float) -> tuple[NDArrayFloat, NDArrayFloat]:

    """
    Rotates points (x, y) by angle (radians).
    x: array of x coords
    y: array of y coords
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _check_equal_size(x, y)
    res_x = np.empty_like(x)
    res_y = np.empty_like(y)
    
    _wrap("prime_rotate_2d", [c_n, c_p, c_p, c_p, c_p, ctypes.c_double])(
        x.size, 
        res_x.ctypes.data_as(c_p), 
        res_y.ctypes.data_as(c_p),
        x.ctypes.data_as(c_p),
        y.ctypes.data_as(c_p),
        ctypes.c_double(angle_rad))
    return res_x, res_y
