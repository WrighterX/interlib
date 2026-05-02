/// Lagrange Interpolation Module (Barycentric Form)
///
/// Thin PyO3 wrapper over the shared `LagrangeCore`.
use crate::core::lagrange_core::LagrangeCore;
use crate::python::pywrap_macros::py_call_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct LagrangeInterpolator {
    core: LagrangeCore,
}

#[pymethods]
impl LagrangeInterpolator {
    #[new]
    pub fn new() -> Self {
        LagrangeInterpolator {
            core: LagrangeCore::new(),
        }
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.core.fit(x, y).map_err(PyValueError::new_err)
    }

    pub fn update_y(&mut self, y: Vec<f64>) -> PyResult<()> {
        self.core.update_y(y).map_err(PyValueError::new_err)
    }

    pub fn add_point(&mut self, x_new: f64, y_new: f64) -> PyResult<()> {
        self.core
            .add_point(x_new, y_new)
            .map_err(PyValueError::new_err)
    }

    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        py_call_impl(&self.core, py, x)
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
