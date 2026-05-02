/// Least Squares Polynomial Approximation module reusing shared core.
use crate::core::least_squares_core::LeastSquaresCore;
use crate::python::pywrap_macros::py_call_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct LeastSquaresInterpolator {
    core: LeastSquaresCore,
}

#[pymethods]
impl LeastSquaresInterpolator {
    #[new]
    #[pyo3(signature = (degree=2))]
    pub fn new(degree: usize) -> Self {
        LeastSquaresInterpolator {
            core: LeastSquaresCore::new(degree),
        }
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.core.fit(x, y).map_err(PyValueError::new_err)
    }

    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        self.core.get_coefficients().map_err(PyValueError::new_err)
    }

    pub fn get_degree(&self) -> usize {
        self.core.degree()
    }

    pub fn r_squared(&self) -> PyResult<f64> {
        self.core.r_squared().map_err(PyValueError::new_err)
    }

    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        py_call_impl(&self.core, py, x)
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
