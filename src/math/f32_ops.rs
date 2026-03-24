use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn prime_sin_f32<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let result: Vec<f32> = x.as_slice()?.par_iter().map(|&a| a.sin()).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_cos_f32<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let result: Vec<f32> = x.as_slice()?.par_iter().map(|&a| a.cos()).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_tan_f32<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let result: Vec<f32> = x.as_slice()?.par_iter().map(|&a| a.tan()).collect();
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_dot_f32(x: PyReadonlyArray1<f32>, y: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }
    Ok(xs.par_iter().zip(ys.par_iter()).map(|(&a, &b)| a * b).sum())
}

#[pyfunction]
pub fn prime_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let a_s = a.as_array();
    let b_s = b.as_array();

    let (m, k) = (a_s.nrows(), a_s.ncols());
    let (k2, n) = (b_s.nrows(), b_s.ncols());

    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Shape mismatch: ({m}, {k}) @ ({k2}, {n}) is invalid"),
        ));
    }

    let mut flat_result = vec![0.0_f32; m * n];
    flat_result
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                row[j] = (0..k).map(|p| a_s[[i, p]] * b_s[[p, j]]).sum();
            }
        });

    let result = numpy::ndarray::Array2::from_shape_vec((m, n), flat_result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub fn prime_rotate_2d_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
    angle_rad: f32,
) -> PyResult<(Bound<'py, numpy::PyArray1<f32>>, Bound<'py, numpy::PyArray1<f32>>)> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Array size mismatch"));
    }

    let c = angle_rad.cos();
    let s = angle_rad.sin();

    let (res_x, res_y): (Vec<f32>, Vec<f32>) = xs
        .par_iter()
        .zip(ys.par_iter())
        .map(|(&px, &py_val)| (px * c - py_val * s, px * s + py_val * c))
        .unzip();

    Ok((res_x.into_pyarray(py), res_y.into_pyarray(py)))
}
