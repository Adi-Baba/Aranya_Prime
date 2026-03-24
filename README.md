<div align="center">

# Aranya Prime

**High-Performance Numerical Computing for Python, Powered by Rust**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build](https://img.shields.io/badge/build-maturin-blueviolet.svg)](https://github.com/PyO3/maturin)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.20-013243.svg)](https://numpy.org/)

[Features](#features) · [Installation](#installation) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Benchmarks](#benchmarks) · [Contributing](#contributing)

</div>

---

Aranya Prime is a drop-in acceleration library for Python that replaces performance-critical NumPy operations with parallel Rust kernels. It provides **zero-copy interop** with NumPy arrays through [PyO3](https://pyo3.rs/) and [rust-numpy](https://github.com/PyO3/rust-numpy), automatic **BLAS/LAPACK dispatch** for large workloads, and SIMD-accelerated signal processing — all while preserving Rust's memory safety guarantees.

## Features

| Category | Capabilities |
|:---|:---|
| **Math & Statistics** | Element-wise arithmetic, parallel reductions (`sum`, `mean`, `std`), `clip`, `l2_norm`, `linf_norm`, polynomial evaluation |
| **Trigonometry** | `sin`, `cos`, `tan` — parallelized across cores via [Rayon](https://github.com/rayon-rs/rayon) |
| **Linear Algebra** | `dot`, `matmul`, `normalize`, `normalize_batch`, `svd` — with auto-dispatch to Fortran BLAS/LAPACK for large matrices |
| **Signal Processing** | `fft` / `ifft` ([RustFFT](https://github.com/ejmahler/RustFFT)), `dct` (DCT-II, orthonormal), `wavelet_transform` (Haar) |
| **Transforms** | `scale`, `rotate_2d` — fused kernels computing both outputs in a single parallel pass |
| **f32 Fast-Math** | Single-precision variants (`f32.sin`, `f32.matmul`, …) for higher throughput on SIMD hardware |
| **Streaming** | Cache-optimized chunked kernels (`stream.sin`, `stream.rotate_2d`) with configurable chunk sizes for datasets exceeding L3 cache |

### Architecture

```
Python  ──►  aranya_prime (Python API)
                 │
                 ▼
            PyO3 FFI Layer  ◄──►  NumPy Buffers (zero-copy)
                 │
          ┌──────┴──────┐
          ▼              ▼
   Rayon Thread Pool   BLAS/LAPACK
   (parallel Rust      (Fortran dgemm,
    kernels, SIMD)      dgesvd, ddot)
```

**Smart dispatch** — Operations automatically route to the optimal backend:
- `dot()` with >1M elements → BLAS `ddot`
- `matmul()` with >500K multiply-adds → BLAS `dgemm`
- Smaller workloads stay on Rayon to avoid Fortran call overhead

## Installation

### Prerequisites

- Python 3.8+
- [Rust toolchain](https://rustup.rs/) (1.70+)
- OpenBLAS development headers (for BLAS/LAPACK support)

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

# Production install
pip install .

# Development install (debug build, faster compile)
pip install maturin
maturin develop
```

## Quick Start

```python
import numpy as np
import aranya_prime as ap

x = np.random.randn(1_000_000)
y = np.random.randn(1_000_000)

# ── Arithmetic ──────────────────────────────────
result = ap.add(x, y)          # element-wise addition
product = ap.mul(x, y)         # element-wise multiplication

# ── Statistics ──────────────────────────────────
avg = ap.mean(x)               # parallel mean
sigma = ap.std(x)              # parallel standard deviation
norm = ap.l2_norm(x)           # Euclidean norm

# ── Linear Algebra ─────────────────────────────
d = ap.dot(x, y)               # auto-dispatches to BLAS at scale

A = np.random.randn(512, 512)
B = np.random.randn(512, 512)
C = ap.matmul(A, B)            # BLAS dgemm for large matrices
U, S, Vh = ap.svd(A)           # LAPACK dgesvd

# ── Signal Processing ──────────────────────────
real, imag = ap.fft(x)         # forward FFT
reconstructed = ap.ifft(real, imag)  # inverse FFT
coeffs = ap.dct(x)             # DCT-II (orthonormal)

# ── f32 Fast-Math ──────────────────────────────
x32 = np.random.randn(1_000_000).astype(np.float32)
fast_sin = ap.f32.sin(x32)     # single-precision, ~2x throughput

# ── Streaming (cache-optimized) ────────────────
result = ap.stream.sin(x, chunk_size=65536)
```

## API Reference

### Core Operations

| Function | Signature | Description |
|:---|:---|:---|
| `add(x, y)` | `NDArray → NDArray` | Element-wise addition |
| `sub(x, y)` | `NDArray → NDArray` | Element-wise subtraction |
| `mul(x, y)` | `NDArray → NDArray` | Element-wise multiplication |
| `div(x, y)` | `NDArray → NDArray` | Element-wise division |
| `polynomial(x)` | `NDArray → NDArray` | Parallel evaluation of x³ + x² + x |
| `clip(x, lo, hi)` | `NDArray → NDArray` | Element-wise clamping |

### Trigonometry

| Function | Signature | Description |
|:---|:---|:---|
| `sin(x)` | `NDArray → NDArray` | Parallel sine |
| `cos(x)` | `NDArray → NDArray` | Parallel cosine |
| `tan(x)` | `NDArray → NDArray` | Parallel tangent |

### Statistics & Norms

| Function | Signature | Description |
|:---|:---|:---|
| `sum(x)` | `NDArray → float` | Parallel sum |
| `mean(x)` | `NDArray → float` | Parallel mean |
| `std(x)` | `NDArray → float` | Parallel standard deviation |
| `l2_norm(x)` | `NDArray → float` | Euclidean norm |
| `linf_norm(x)` | `NDArray → float` | L∞ norm (max absolute value) |

### Linear Algebra

| Function | Signature | Description |
|:---|:---|:---|
| `dot(x, y)` | `NDArray, NDArray → float` | Dot product (auto BLAS dispatch) |
| `matmul(A, B)` | `2D, 2D → 2D` | Matrix multiplication (auto BLAS dispatch) |
| `normalize(x)` | `NDArray → NDArray` | Unit vector normalization |
| `normalize_batch(X)` | `2D → 2D` | Row-wise L2 normalization |
| `svd(A)` | `2D → (U, S, Vh)` | Singular Value Decomposition (LAPACK) |

### Signal Processing

| Function | Signature | Description |
|:---|:---|:---|
| `fft(x)` | `NDArray → (real, imag)` | Forward FFT |
| `ifft(re, im)` | `NDArray, NDArray → NDArray` | Inverse FFT |
| `dct(x)` | `NDArray → NDArray` | DCT-II (orthonormal) |
| `wavelet_transform(x)` | `NDArray → NDArray` | Haar wavelet (single-level) |
| `convolve(signal, kernel)` | `NDArray, NDArray → NDArray` | 1D direct convolution (full mode) |

### Transforms

| Function | Signature | Description |
|:---|:---|:---|
| `scale(x, factor)` | `NDArray → NDArray` | Parallel scalar multiplication |
| `rotate_2d(x, y, angle)` | `NDArray, NDArray, float → (NDArray, NDArray)` | Fused 2D rotation |

### f32 Fast-Math (`ap.f32.*`)

Single-precision variants for higher throughput: `sin`, `cos`, `tan`, `dot`, `matmul`, `rotate_2d`

### Streaming (`ap.stream.*`)

Cache-optimized chunked processing with configurable `chunk_size` (default: 65536): `sin`, `rotate_2d`

## Benchmarks

Performance comparison against NumPy on representative workloads (measured via `pytest-benchmark`):

| Operation | Size | vs. NumPy |
|:---|:---|:---|
| `dot` | 1M elements | ~1.5–2.5x faster |
| `matmul` | 512×512 | ~1.4–2.1x faster |
| `fft` | 1M elements | ~1.2–1.5x faster |
| `svd` | 256×256 | ~1.0–1.3x faster |
| `dct` | 1M elements | ~1.2–1.8x faster |

> Results vary by hardware, BLAS backend, and core count. Run benchmarks on your machine for accurate numbers.

```bash
# Run full benchmark suite
pytest tests/test_benchmarks.py --benchmark-only --benchmark-group-by=group

# Run correctness tests
pytest tests/test_correctness.py tests/test_f32.py tests/test_streaming.py -v
```

## Project Structure

```
├── src/                    # Rust source
│   ├── lib.rs              # PyO3 module registration
│   ├── math/               # Arithmetic, trig, stats, FFT, DCT, wavelets, f32, streaming
│   ├── linalg/             # Dot, matmul, normalize, BLAS/LAPACK bridge, SVD
│   └── transform/          # Scale, rotate_2d
├── python/
│   └── aranya_prime/       # Python package (API wrappers, type stubs)
├── tests/                  # pytest suite (correctness, f32, benchmarks, streaming)
├── Cargo.toml              # Rust dependencies & release profile
└── pyproject.toml          # Python build config (maturin)
```

## Roadmap

- [x] Core mathematical kernels & parallel statistics
- [x] Signal processing — FFT, DCT-II, Haar wavelet
- [x] BLAS/LAPACK bridge with smart dispatch
- [x] f32 fast-math variants
- [x] Streaming / cache-optimized chunked kernels
- [ ] CI/CD — multi-platform wheel builds (Linux, macOS Intel/ARM)
- [ ] PyPI distribution via `maturin publish`
- [ ] Eigenvalue/eigenvector solvers
- [ ] Sparse matrix support
- [ ] Window functions & FIR/IIR filters

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/Adi-Baba/Aranya_Prime.git
cd Aranya_Prime
python -m venv venv && source venv/bin/activate
pip install maturin numpy pytest pytest-benchmark
maturin develop
pytest -v
```

## License

MIT — see [LICENSE](LICENSE) for details.
