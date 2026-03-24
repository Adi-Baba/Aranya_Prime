use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn prime_math_sum<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    let result: Vec<f64> = xs.par_iter().zip(ys.par_iter()).map(|(&a, &b)| a + b).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_sub<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    let result: Vec<f64> = xs.par_iter().zip(ys.par_iter()).map(|(&a, &b)| a - b).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_mul<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    let result: Vec<f64> = xs.par_iter().zip(ys.par_iter()).map(|(&a, &b)| a * b).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_div<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    // f64 division by zero yields Inf/NaN exactly like NumPy — intentional.
    let result: Vec<f64> = xs.par_iter().zip(ys.par_iter()).map(|(&a, &b)| a / b).collect();
    Ok(result.into_pyarray(py))
}
