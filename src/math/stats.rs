use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// L2 norm: sqrt(sum(x^2)) — Euclidean length of the vector.
#[pyfunction]
pub fn prime_l2_norm(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    Ok(xs.par_iter().map(|&a| a * a).sum::<f64>().sqrt())
}

/// L∞ norm: max(|x|) — the largest absolute value in the vector.
#[pyfunction]
pub fn prime_linf_norm(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    // fold_with is the Rayon equivalent of a parallel max
    let max = xs.par_iter()
        .map(|&a| a.abs())
        .reduce(|| 0.0_f64, f64::max);
    Ok(max)
}

/// Parallel sum of all elements.
#[pyfunction]
pub fn prime_sum(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(x.as_slice()?.par_iter().sum())
}

/// Parallel mean (average) of all elements.
#[pyfunction]
pub fn prime_mean(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    if xs.is_empty() {
        return Ok(f64::NAN);
    }
    let s: f64 = xs.par_iter().sum();
    Ok(s / xs.len() as f64)
}

/// Standard deviation using a two-pass parallel algorithm (numerically stable).
#[pyfunction]
pub fn prime_std(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    if xs.len() < 2 {
        return Ok(0.0);
    }
    let n = xs.len() as f64;
    let mean: f64 = xs.par_iter().sum::<f64>() / n;
    let variance: f64 = xs.par_iter().map(|&a| (a - mean).powi(2)).sum::<f64>() / n;
    Ok(variance.sqrt())
}

/// Clamps every element to [min_val, max_val].
#[pyfunction]
pub fn prime_clip<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    min_val: f64,
    max_val: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result: Vec<f64> = x.as_slice()?
        .par_iter()
        .map(|&a| a.clamp(min_val, max_val))
        .collect();
    Ok(result.into_pyarray(py))
}
