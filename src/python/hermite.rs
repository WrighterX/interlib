/// Hermite Interpolation Module
///
/// This module implements Hermite polynomial interpolation, which constructs
/// a polynomial matching both function values AND derivative values at data points.
///
/// The interpolation is computed using divided differences with doubled points
/// and is exposed to Python via PyO3.
use crate::core::hermite_core::HermiteCore;
use crate::python::pywrap_macros::py_call_impl;
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
        py_call_impl(&self.core, py, x)
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
