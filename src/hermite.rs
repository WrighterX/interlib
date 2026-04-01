/// Hermite Interpolation Module
///
/// This module implements Hermite polynomial interpolation, which constructs
/// a polynomial matching both function values AND derivative values at data points.
///
/// The interpolation is computed using divided differences with doubled points
/// and is exposed to Python via PyO3.

use crate::hermite_core::HermiteCore;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Hermite Polynomial Interpolator
#[pyclass]
pub struct HermiteInterpolator {
    core: HermiteCore,
}

#[pymethods]
impl HermiteInterpolator {
    #[new]
    pub fn new() -> Self {
        HermiteInterpolator {
            core: HermiteCore::new(),
        }
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>, dy: Vec<f64>) -> PyResult<()> {
        self.core.fit(x, y, dy).map_err(PyValueError::new_err)
    }

    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        self.core.get_coefficients().map_err(PyValueError::new_err)
    }

    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(single_x) = x.extract::<f64>() {
            let result = self.core.evaluate_single(single_x).map_err(PyValueError::new_err)?;
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let x_slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
            {
                let result_slice = unsafe { result_array.as_slice_mut()? };
                self.core
                    .fill_many(x_slice, result_slice)
                    .map_err(PyValueError::new_err)?;
            }
            return Ok(result_array.into_any().unbind());
        }

        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results = self.core.evaluate_many(&x_list).map_err(PyValueError::new_err)?;
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, list of floats, or NumPy array",
        ))
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
