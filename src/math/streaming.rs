use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// A block-based parallel sin implementation.
/// Processes the array in contiguous chunks of `chunk_size`.
/// This can improve cache locality for very large arrays.
#[pyfunction]
pub fn prime_chunked_sin<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    chunk_size: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let mut result = vec![0.0; xs.len()];

    // Use par_chunks_mut to process blocks in parallel.
    // Within each block, we iterate sequentially to keep the data in L1/L2 cache.
    result.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let start = i * chunk_size;
            for (j, val) in chunk.iter_mut().enumerate() {
                *val = xs[start + j].sin();
            }
        });

    Ok(result.into_pyarray(py))
}

/// Block-based parallel 2D rotation.
/// Fuses sin/cos calculation per chunk and maximizes cache hits.
#[pyfunction]
pub fn prime_chunked_rotate_2d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    angle_rad: f64,
    chunk_size: usize,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }

    let mut res_x = vec![0.0; xs.len()];
    let mut res_y = vec![0.0; xs.len()];

    let (c, s) = (angle_rad.cos(), angle_rad.sin());

    // Iterate over chunks of both arrays simultaneously
    res_x.par_chunks_mut(chunk_size)
        .zip(res_y.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(i, (cx, cy))| {
            let start = i * chunk_size;
            for j in 0..cx.len() {
                let px = xs[start + j];
                let py_val = ys[start + j];
                cx[j] = px * c - py_val * s;
                cy[j] = px * s + py_val * c;
            }
        });

    Ok((res_x.into_pyarray(py), res_y.into_pyarray(py)))
}
