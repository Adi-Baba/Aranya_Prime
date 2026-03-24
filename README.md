<div align="center">

# Aranya Prime

**Numerical Computing for Python with Rust**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build](https://img.shields.io/badge/build-maturin-blueviolet.svg)](https://github.com/PyO3/maturin)

[Overview](#overview) · [Installation](#installation) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Benchmarks](#benchmarks) · [Contributing](#contributing)

</div>

---

Aranya Prime provides numerical operations for Python with a Rust implementation. It offers zero-copy NumPy array interop through [PyO3](https://pyo3.rs/), optional BLAS/LAPACK dispatch for large matrices, and signal processing kernels.

## Overview

### When to Use Aranya Prime

**Competitive or faster than NumPy:**
- Matrix multiplication (`matmul`) — comparable performance
- DCT (Discrete Cosine Transform) — slightly faster
- Haar wavelet transform — faster

**Slower than NumPy (use NumPy instead):**
- Dot products — ~20% slower
- FFT — ~25% slower
- SVD — ~3× slower
- Simple reductions (sum, mean) — slower

**Why use it:**
- You need Rust integration in your Python project
- You want to experiment with or extend the Rust kernels
- Educational purposes — learning PyO3 and numerical Rust

### Architecture

```
Python  ──►  aranya_prime (Python API)
                 │
                 ▼
            PyO3 FFI Layer  ◄──►  NumPy Arrays (zero-copy)
                 │
          ┌──────┴──────┐
          ▼              ▼
   Rayon (parallel)   BLAS/LAPACK
   Rust kernels       (OpenBLAS)
```

Operations dispatch based on size:
- `dot()` with >1M elements → BLAS `ddot`
- `matmul()` with >500K multiply-adds → BLAS `dgemm`
- Smaller workloads → Rayon parallel Rust kernels

## Installation

### Prerequisites

- Python 3.8+
- [Rust toolchain](https://rustup.rs/) (1.70+)
- OpenBLAS development headers

```bash
# Ubuntu / Debian
sudo apt install libopenblas-dev

# macOS
brew install openblas
```

### Install from Source

```bash
git clone https://github.com/Adi-Baba/Aranya_Prime.git
cd Aranya_Prime

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install build tools
pip install --upgrade pip maturin

# Build (release mode for best performance)
maturin develop --release
```

### Verify Installation

```bash
pytest tests/test_correctness.py -v
```

## Quick Start

```python
import numpy as np
import aranya_prime as ap

x = np.random.randn(1_000_000)
y = np.random.randn(1_000_000)

# Arithmetic
result = ap.add(x, y)
product = ap.mul(x, y)

# Statistics
avg = ap.mean(x)
norm = ap.l2_norm(x)

# Linear algebra
d = ap.dot(x, y)              # Uses BLAS for large arrays

A = np.random.randn(512, 512)
B = np.random.randn(512, 512)
C = ap.matmul(A, B)           # Competitive with NumPy

# Signal processing
coeffs = ap.dct(x)            # Faster than NumPy
wavelet = ap.wavelet_transform(x)  # Haar wavelet

# Single-precision variants (faster, less precision)
x32 = x.astype(np.float32)
fast_result = ap.f32.sin(x32)
```

## API Reference

### Core Operations

| Function | Description |
|:---|:---|
| `add(x, y)` | Element-wise addition |
| `sub(x, y)` | Element-wise subtraction |
| `mul(x, y)` | Element-wise multiplication |
| `div(x, y)` | Element-wise division |
| `polynomial(x)` | Evaluates x³ + x² + x |
| `clip(x, lo, hi)` | Element-wise clamping |

### Trigonometry

| Function | Description |
|:---|:---|
| `sin(x)` | Sine |
| `cos(x)` | Cosine |
| `tan(x)` | Tangent |

### Statistics & Norms

| Function | Description |
|:---|:---|
| `sum(x)` | Sum of elements |
| `mean(x)` | Mean |
| `std(x)` | Standard deviation |
| `l2_norm(x)` | Euclidean norm |
| `linf_norm(x)` | Max absolute value |

### Linear Algebra

| Function | Description |
|:---|:---|
| `dot(x, y)` | Dot product |
| `matmul(A, B)` | Matrix multiplication |
| `normalize(x)` | Unit vector |
| `normalize_batch(X)` | Row-wise normalization |
| `svd(A)` | SVD via LAPACK |

### Signal Processing

| Function | Description |
|:---|:---|
| `fft(x)` | Forward FFT → `(real, imag)` |
| `ifft(re, im)` | Inverse FFT |
| `dct(x)` | DCT-II (orthonormal) |
| `wavelet_transform(x)` | Haar wavelet |
| `convolve(signal, kernel)` | 1D convolution |

### Transforms

| Function | Description |
|:---|:---|
| `scale(x, factor)` | Scalar multiplication |
| `rotate_2d(x, y, angle)` | 2D rotation |

### f32 Namespace

Single-precision variants: `ap.f32.sin`, `ap.f32.dot`, `ap.f32.matmul`, etc.

### Stream Namespace

Chunked processing for large arrays: `ap.stream.sin`, `ap.stream.rotate_2d`

## Benchmarks

Measured on Linux (Python 3.12, OpenBLAS, 8-core). Lower is better.

### Summary

| Operation | Winner | Notes |
|:---|:---|:---|
| `dot` (1M) | **NumPy** | Aranya ~20% slower |
| `matmul` (512×512) | **Aranya** | Slightly faster than NumPy |
| `fft` (1M) | **NumPy** | Aranya ~25% slower |
| `svd` (256×256) | **NumPy** | Aranya ~3× slower |
| `dct` (1M) | **Aranya** | ~8% faster |
| `wavelet` (1M) | **Aranya** | ~27% faster |

### Detailed Results

**Dot Product (1M elements)**
| Method | Time (μs) | Relative |
|:---|---:|---:|
| NumPy | 397 | 1.0× |
| Aranya (BLAS) | 486 | 1.22× |
| Aranya (Rayon) | 475 | 1.20× |
| Numba | 1,198 | 3.02× |

**Matrix Multiplication**
| Method | Time (ms) | Relative |
|:---|---:|---:|
| Aranya (Rayon) | 3.72 | 1.0× |
| Aranya (BLAS) | 3.80 | 1.02× |
| NumPy | 4.57 | 1.23× |
| Numba | 371.05 | 99.76× |

**FFT (1M elements)**
| Method | Time (ms) | Relative |
|:---|---:|---:|
| NumPy | 38.83 | 1.0× |
| Aranya | 49.04 | 1.26× |
| Numba | 489.18 | 12.60× |

**DCT (1M elements)**
| Method | Time (ms) | Relative |
|:---|---:|---:|
| Aranya | 120.34 | 1.0× |
| NumPy | 129.64 | 1.08× |

**Wavelet Transform (1M elements)**
| Method | Time (ms) | Relative |
|:---|---:|---:|
| Aranya | 1.77 | 1.0× |
| Python/NumPy | 2.24 | 1.27× |

**SVD (256×256)**
| Method | Time (ms) | Relative |
|:---|---:|---:|
| NumPy | 26.08 | 1.0× |
| Aranya | 79.05 | 3.03× |

Run benchmarks on your machine:

```bash
pytest tests/test_benchmarks.py --benchmark-group-by=group -v
```

## Project Structure

```
├── src/
│   ├── lib.rs              # PyO3 module
│   ├── math/               # Arithmetic, trig, FFT, DCT, wavelets
│   ├── linalg/             # Dot, matmul, SVD, BLAS bridge
│   └── transform/          # Scale, rotate
├── python/aranya_prime/    # Python API
├── tests/                  # pytest suite
├── Cargo.toml
└── pyproject.toml
```

## Roadmap

- [x] Core math kernels
- [x] Signal processing (FFT, DCT, wavelet)
- [x] BLAS/LAPACK integration
- [x] f32 variants
- [x] Streaming kernels
- [ ] Pre-built wheels (PyPI)
- [ ] More linear algebra (eigendecomposition)
- [ ] Sparse matrix support

## Contributing

```bash
git clone https://github.com/Adi-Baba/Aranya_Prime.git
cd Aranya_Prime
python -m venv venv && source venv/bin/activate
pip install --upgrade pip maturin numpy pytest pytest-benchmark
maturin develop
pytest -v
```

## License

MIT — see [LICENSE](LICENSE) for details.
