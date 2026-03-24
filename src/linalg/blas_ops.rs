use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::SVDInto;

/// BLAS-accelerated Dot Product.
/// Uses the system's optimized BLAS (OpenBLAS/MKL) for large vector reduction.
#[pyfunction]
pub fn prime_blas_dot(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    
    if x_arr.len() != y_arr.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    
    // ndarray automatically uses BLAS if the feature is enabled
    Ok(x_arr.dot(&y_arr))
}

/// BLAS-accelerated Matrix Multiplication.
/// Uses dgemm from the underlying BLAS library.
#[pyfunction]
pub fn prime_blas_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    
    if a_arr.ncols() != b_arr.nrows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Shape mismatch for matmul"));
    }
    
    let result = a_arr.dot(&b_arr);
    Ok(result.into_pyarray(py))
}

/// LAPACK-accelerated SVD (Singular Value Decomposition).
/// Returns (U, S, Vh) where A = U * S * Vh.
#[pyfunction]
pub fn prime_svd<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
)> {
    // Convert view into an owned matrix so we can take ownership during SVD.
    let a_owned = a.as_array().to_owned();

    // Use `svd_into` to avoid an extra clone/copy when decomposing.
    let (u_opt, s, vt_opt): (Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>) =
        a_owned
            .svd_into(true, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let u = u_opt
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "SVD did not compute U"))?;
    let vt = vt_opt
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "SVD did not compute V^T"))?;

    let u_out = u.into_pyarray(py);
    let s_out = s.into_pyarray(py);
    let vt_out = vt.into_pyarray(py);

    Ok((u_out, s_out, vt_out))
}
