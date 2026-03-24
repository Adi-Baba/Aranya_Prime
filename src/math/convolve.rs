use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// 1D discrete convolution of `signal` with `kernel` (full mode).
///
/// Output length = signal.len() + kernel.len() - 1.
/// This is a direct O(n*k) implementation — fast for small kernels.
/// For large kernels, use `prime_fft_convolve` instead (not yet implemented).
#[pyfunction]
pub fn prime_convolve<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    kernel: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let sig = signal.as_slice()?;
    let ker = kernel.as_slice()?;

    if sig.is_empty() || ker.is_empty() {
        return Ok(vec![].into_pyarray(py));
    }

    let out_len = sig.len() + ker.len() - 1;

    // Each output index is independent — embarrassingly parallel.
    let result: Vec<f64> = (0..out_len)
        .into_par_iter()
        .map(|i| {
            let kstart = if i + 1 >= ker.len() { i + 1 - ker.len() } else { 0 };
            let kend = i.min(sig.len() - 1);
            (kstart..=kend)
                .map(|j| sig[j] * ker[i - j])
                .sum()
        })
        .collect();

    Ok(result.into_pyarray(py))
}
