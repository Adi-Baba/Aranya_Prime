use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Evaluates x³ + x² + x element-wise using Rayon parallelism.
#[pyfunction]
pub fn prime_poly<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result: Vec<f64> = x.as_slice()?
        .par_iter()
        .map(|&a| (a * a * a) + (a * a) + a)
        .collect();
    Ok(result.into_pyarray(py))
}
