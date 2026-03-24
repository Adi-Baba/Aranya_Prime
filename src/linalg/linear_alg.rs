use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Computes the dot product using a parallel reduction.
#[pyfunction]
pub fn prime_dot(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    let result: f64 = xs.par_iter().zip(ys.par_iter()).map(|(&a, &b)| a * b).sum();
    Ok(result)
}

/// Computes the Euclidean norm (magnitude) of a vector.
#[pyfunction]
pub fn prime_mag(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let sum_sq: f64 = xs.par_iter().map(|&a| a * a).sum();
    Ok(sum_sq.sqrt())
}

/// Returns a normalized (unit) version of the input vector.
#[pyfunction]
pub fn prime_normalize<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let mag = xs.par_iter().map(|&a| a * a).sum::<f64>().sqrt();

    let result: Vec<f64> = if mag == 0.0 {
        vec![0.0; xs.len()]
    } else {
        xs.par_iter().map(|&a| a / mag).collect()
    };

    Ok(result.into_pyarray(py))
}
