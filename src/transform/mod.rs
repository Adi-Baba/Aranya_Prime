use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Multiplies every element of `x` by scalar `s`.
#[pyfunction]
pub fn prime_scale<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    s: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let result: Vec<f64> = x.as_slice()?.par_iter().map(|&a| a * s).collect();
    Ok(result.into_pyarray(py))
}

/// Rotates 2D point arrays (x, y) by `angle_rad` radians.
///
/// This is a fused kernel: both output arrays are computed in a single
/// parallel pass, avoiding the multiple memory sweeps NumPy would require.
#[pyfunction]
pub fn prime_rotate_2d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    angle_rad: f64,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }

    let c = angle_rad.cos();
    let s = angle_rad.sin();

    // Rayon unzip computes both new_x and new_y simultaneously in one pass.
    let (res_x, res_y): (Vec<f64>, Vec<f64>) = xs
        .par_iter()
        .zip(ys.par_iter())
        .map(|(&px, &py_val)| (px * c - py_val * s, px * s + py_val * c))
        .unzip();

    Ok((res_x.into_pyarray(py), res_y.into_pyarray(py)))
}
