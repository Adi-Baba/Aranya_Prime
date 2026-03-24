use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Matrix multiplication: C = A @ B.
///
/// A is (m, k), B is (k, n), C is (m, n).
/// Uses BLAS (via ndarray's optimized dot product) for all sizes.
/// ndarray-linalg with OpenBLAS provides performance comparable to NumPy.
#[pyfunction]
pub fn prime_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let a_s = a.as_array();
    let b_s = b.as_array();

    let (m, k) = (a_s.nrows(), a_s.ncols());
    let (k2, n) = (b_s.nrows(), b_s.ncols());

    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Shape mismatch: ({m}, {k}) @ ({k2}, {n}) is invalid"),
        ));
    }

    // Use BLAS-accelerated dot product (ndarray-linalg with OpenBLAS)
    let result = a_s.dot(&b_s);
    Ok(result.into_pyarray(py))
}

/// Row-wise L2 normalization of a 2D matrix.
///
/// Each row of X is divided by its L2 norm. Rows are processed in parallel.
#[pyfunction]
pub fn prime_normalize_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let x_arr = x.as_array();
    let (m, n) = (x_arr.nrows(), x_arr.ncols());

    let mut flat: Vec<f64> = x_arr.iter().cloned().collect();

    flat.par_chunks_mut(n).for_each(|row| {
        let norm = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
    });

    let result = numpy::ndarray::Array2::from_shape_vec((m, n), flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py))
}
