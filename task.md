# Aranya Prime — Phase 1: Harden the Core ✅ COMPLETE

## Bug Fixes
- [x] Fix `RuntimeWarning` in [test_chaos_handling](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#20-40)
- [x] Keep `Cargo.lock` committed for reproducibility, add `target/` + `venv/` to [.gitignore](file:///home/aditya/Documents/Math/AranyaP/.gitignore)

## 2D Array Support
- [x] [matmul(A, B)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#62-65) — row-parallel matrix multiplication (`PyReadonlyArray2`)
- [x] [normalize_batch(X)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#66-69) — row-wise L2 normalization of 2D matrix

## New 1D Kernels
- [x] [l2_norm(x)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#45-46) — Euclidean norm
- [x] [linf_norm(x)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#46-47) — L∞ norm: max(|x|)
- [x] [convolve(signal, kernel)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#80-83) — parallel direct 1D convolution
- [x] [sum(x)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#41-42), [mean(x)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#42-43), [std(x)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#43-44) — parallel reductions
- [x] [clip(x, min, max)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#44-45) — element-wise clamping

## FFT Kernel
- [x] Add `rustfft` crate (SIMD AVX/SSE/NEON accelerated)
- [x] [fft(x)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#84-87) → [(real, imag)](file:///home/aditya/Documents/Math/AranyaP/aranya_prime/core.py#79-85) tuple matching `np.fft.fft`
- [x] [ifft(re, im)](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#88-91) → reconstructed real signal with 1/N normalization

## Testing — 21/21 passing, 0 warnings
- [x] [test_statistics](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#119-128), [test_norms](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#129-134), [test_convolve](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#135-142)
- [x] [test_fft_roundtrip](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#143-149), [test_fft_matches_numpy](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#150-157)
- [x] [test_matmul](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#158-165), [test_normalize_batch](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_correctness.py#166-172)

---

## Phase 2: Ship It (next)
- [ ] Set up GitHub Actions to build binary wheels (Linux, macOS Intel, macOS ARM)
- [ ] Publish to PyPI via `maturin publish`

## Phase 3: Differentiate (active)

### f32 Fast-Math Variants ✅
- [x] Add [f32](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/tests/test_f32.py#48-51) versions of [sin](file:///home/aditya/Documents/Math/AranyaP/aranya_prime/core.py#79-85), [cos](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#31-32), [dot](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#49-52), [matmul](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#62-65), [rotate_2d](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/python/aranya_prime/__init__.py#105-107) in [src/math/f32_ops.rs](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/src/math/f32_ops.rs)
- [x] Expose as `ap.f32.sin(x)`, `ap.f32.matmul(A,B)` etc. via a sub-namespace in [__init__.py](file:///home/aditya/Documents/Math/AranyaP/aranya_prime/__init__.py)
- [x] Add correctness tests (atol=1e-6) and perf benchmarks vs f64 variants

### BLAS/LAPACK Bridge (Fortran FFI) (active)
- [x] Add `blas-src` and `lapack-src` to [Cargo.toml](file:///home/aditya/Documents/Math/AranyaP/AranyaP_Rust/Cargo.toml)
- [x] `blas_matmul(A, B)` — delegate to `dgemm` for large matrices (via ndarray/BLAS)
- [x] `blas_dot(x, y)` — delegate to `ddot`
- [x] `svd(A)` — Singular Value Decomposition via `dgesvd` (LAPACK)
- [x] Add auto-dispatch: small matrices use Rayon kernel, large use BLAS
- [x] Add correctness tests vs `np.linalg.svd` and `np.dot`

#### Next steps for production readiness
- [ ] Add CI workflows to build & test wheels (Linux/macOS Intel/macOS ARM)
- [ ] Add release pipeline: `maturin publish` or GitHub Actions drop-in
- [ ] Document performance tuning knobs (BLAS threshold, thread count, `OMP_NUM_THREADS`)
- [ ] Add “profiling mode” build that emits call timings / pprof stats
- [ ] Add a stable public API section to README (example usage + benchmark notes)


### Phase 4: Advanced Development & Optimization (planned)

- [ ] Add new mathematical or scientific libraries
	- [x] Discrete Cosine Transform (DCT)
	- [ ] Wavelet Transform
	- [ ] Eigenvalue/Eigenvector solvers
	- [ ] Polynomial root finding and interpolation
	- [ ] Statistical distributions (PDF, CDF, sampling)
	- [ ] Signal processing tools (filters, window functions)
	- [ ] Sparse matrix support
- [ ] Profile and optimize existing Rust and Python code for speed and memory usage
- [ ] Implement SIMD/vectorization in Rust for critical kernels
- [ ] Expand BLAS/LAPACK coverage (e.g., add more routines, support for sparse matrices)
- [ ] Improve Python API ergonomics and documentation
- [ ] Add more benchmarks and real-world use-case tests
- [ ] Package as a standalone library for easy pip install and conda support
- [ ] Set up CI for multi-platform wheels and automated testing
- [ ] Write user and developer documentation
