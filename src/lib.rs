extern crate blas_src;
extern crate lapack_src;

use pyo3::prelude::*;

mod linalg;
mod math;
mod transform;

#[pymodule]
fn _aranya_prime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Array Operations ───────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::array_ops::prime_math_sum, m)?)?;
    m.add_function(wrap_pyfunction!(math::array_ops::prime_sub, m)?)?;
    m.add_function(wrap_pyfunction!(math::array_ops::prime_mul, m)?)?;
    m.add_function(wrap_pyfunction!(math::array_ops::prime_div, m)?)?;

    // ── Statistics & Norms ────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::stats::prime_sum, m)?)?;
    m.add_function(wrap_pyfunction!(math::stats::prime_mean, m)?)?;
    m.add_function(wrap_pyfunction!(math::stats::prime_std, m)?)?;
    m.add_function(wrap_pyfunction!(math::stats::prime_clip, m)?)?;
    m.add_function(wrap_pyfunction!(math::stats::prime_l2_norm, m)?)?;
    m.add_function(wrap_pyfunction!(math::stats::prime_linf_norm, m)?)?;

    // ── Trigonometry ──────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::trig::prime_sin, m)?)?;
    m.add_function(wrap_pyfunction!(math::trig::prime_cos, m)?)?;
    m.add_function(wrap_pyfunction!(math::trig::prime_tan, m)?)?;

    // ── Polynomials ───────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::poly::prime_poly, m)?)?;

    // ── Convolution ───────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::convolve::prime_convolve, m)?)?;

    // ── FFT ───────────────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::fft::prime_fft, m)?)?;
    m.add_function(wrap_pyfunction!(math::fft::prime_ifft, m)?)?;

    // ── DCT & Wavelet ─────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::dct_wavelet::prime_dct, m)?)?;
    m.add_function(wrap_pyfunction!(math::dct_wavelet::prime_wavelet_transform, m)?)?;

    // ── Linear Algebra (1D) ───────────────────────────────────────────
    m.add_function(wrap_pyfunction!(linalg::linear_alg::prime_dot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linear_alg::prime_mag, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::linear_alg::prime_normalize, m)?)?;

    // ── Linear Algebra (2D) ───────────────────────────────────────────
    m.add_function(wrap_pyfunction!(linalg::matmul::prime_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matmul::prime_normalize_batch, m)?)?;

    // ── BLAS / LAPACK (Fortran FFI) ───────────────────────────────────
    m.add_function(wrap_pyfunction!(linalg::blas_ops::prime_blas_dot, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::blas_ops::prime_blas_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::blas_ops::prime_svd, m)?)?;

    // ── Transforms ────────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(transform::prime_scale, m)?)?;
    m.add_function(wrap_pyfunction!(transform::prime_rotate_2d, m)?)?;

    // ── f32 Fast-Math Variants ─────────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::f32_ops::prime_sin_f32, m)?)?;
    m.add_function(wrap_pyfunction!(math::f32_ops::prime_cos_f32, m)?)?;
    m.add_function(wrap_pyfunction!(math::f32_ops::prime_tan_f32, m)?)?;
    m.add_function(wrap_pyfunction!(math::f32_ops::prime_dot_f32, m)?)?;
    m.add_function(wrap_pyfunction!(math::f32_ops::prime_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(math::f32_ops::prime_rotate_2d_f32, m)?)?;

    // ── Streaming / Chunked Kernels ────────────────────────────────────
    m.add_function(wrap_pyfunction!(math::streaming::prime_chunked_sin, m)?)?;
    m.add_function(wrap_pyfunction!(math::streaming::prime_chunked_rotate_2d, m)?)?;

    Ok(())
}
