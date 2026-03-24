use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn prime_sin<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result: Vec<f64> = x.as_slice()?.par_iter().map(|&a| a.sin()).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_cos<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result: Vec<f64> = x.as_slice()?.par_iter().map(|&a| a.cos()).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_tan<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result: Vec<f64> = x.as_slice()?.par_iter().map(|&a| a.tan()).collect();
    Ok(result.into_pyarray(py))
}
