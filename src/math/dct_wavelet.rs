use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

/// Computes the Discrete Cosine Transform (DCT-II) of a real-valued signal.
/// Returns a 1D array of DCT coefficients (orthonormal / 'ortho' scaling).
///
/// This uses an FFT-based strategy (O(n log n)) instead of the naive O(n²)
/// direct summation, leveraging the identity between DCT-II and a symmetric FFT.
#[pyfunction]
pub fn prime_dct<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }

    // Build a symmetric sequence of length 2n: [x, reverse(x)]
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(2 * n);
    for &v in xs {
        buffer.push(Complex { re: v, im: 0.0 });
    }
    for &v in xs.iter().rev() {
        buffer.push(Complex { re: v, im: 0.0 });
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(buffer.len());
    fft.process(&mut buffer);

    let mut result = vec![0.0; n];
    // For orthonormal DCT-II, we need sqrt(2/n) for k>0 and sqrt(1/n) for k=0.
    // The symmetric FFT produces 2x the expected amplitude, so we divide by 2.
    let sqrt_2_over_n = (2.0 / n as f64).sqrt() / 2.0;
    let sqrt_1_over_n = (1.0 / n as f64).sqrt() / 2.0;
    let two_n = (2 * n) as f64;

    for k in 0..n {
        let phase = -std::f64::consts::PI * k as f64 / two_n;
        let tw = Complex::from_polar(1.0, phase);
        let val = buffer[k] * tw;
        let scale = if k == 0 { sqrt_1_over_n } else { sqrt_2_over_n };
        result[k] = val.re * scale;
    }

    Ok(result.into_pyarray(py))
}

/// Computes the Haar wavelet transform (1D, single-level) of a real-valued signal.
/// Returns a 1D array: [approximation coefficients..., detail coefficients...]
#[pyfunction]
pub fn prime_wavelet_transform<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let n = xs.len();
    if n % 2 != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input length must be even for Haar wavelet transform",
        ));
    }
    let mut result = vec![0.0; n];
    let sqrt2 = std::f64::consts::SQRT_2;
    // Approximation coefficients (low-pass)
    for i in 0..(n / 2) {
        result[i] = (xs[2 * i] + xs[2 * i + 1]) / sqrt2;
    }
    // Detail coefficients (high-pass)
    for i in 0..(n / 2) {
        result[n / 2 + i] = (xs[2 * i] - xs[2 * i + 1]) / sqrt2;
    }
    Ok(result.into_pyarray(py))
}
