use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

/// Computes the FFT of a real-valued signal.
///
/// Returns a tuple of (real_part, imag_part) arrays, matching NumPy's behaviour.
/// Uses `rustfft` — a pure-Rust, SIMD-accelerated FFT library (AVX/SSE/NEON).
#[pyfunction]
pub fn prime_fft<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
    let xs = x.as_slice()?;

    // Convert real input to complex. rustfft works natively with Complex<f64>.
    let mut buffer: Vec<Complex<f64>> = xs.iter().map(|&r| Complex { re: r, im: 0.0 }).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(buffer.len());
    fft.process(&mut buffer);

    let real: Vec<f64> = buffer.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = buffer.iter().map(|c| c.im).collect();

    Ok((real.into_pyarray(py), imag.into_pyarray(py)))
}

/// Computes the inverse FFT from (real, imag) arrays.
///
/// Returns the reconstructed real-valued signal. The output is scaled by 1/N,
/// matching NumPy's `np.fft.ifft` normalization.
#[pyfunction]
pub fn prime_ifft<'py>(
    py: Python<'py>,
    re: PyReadonlyArray1<'py, f64>,
    im: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let res = re.as_slice()?;
    let ims = im.as_slice()?;
    if res.len() != ims.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "real and imag arrays must be the same length",
        ));
    }

    let n = res.len();
    let mut buffer: Vec<Complex<f64>> = res.iter().zip(ims.iter())
        .map(|(&r, &i)| Complex { re: r, im: i })
        .collect();

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    // Scale by 1/N to match NumPy's default normalization
    let scale = 1.0 / n as f64;
    let result: Vec<f64> = buffer.iter().map(|c| c.re * scale).collect();

    Ok(result.into_pyarray(py))
}
