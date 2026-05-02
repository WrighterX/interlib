/// Radial Basis Function (RBF) interpolation for Python plus FFI consumers.
///
/// This module exposes the Python-facing `RBFInterpolator`, which wraps the
/// shared `RBFCore` implementation to keep the interpolation logic consistent
/// between the Python bindings and the MATLAB/FFI layers. Supported kernels are
/// Gaussian, multiquadric, inverse multiquadric, thin plate spline, and linear.
use crate::python::pywrap_macros::py_call_impl;
use crate::core::rbf_core::{RBFCore, RBFKernel};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A thin Python layer around the shared RBF core implementation.
#[pyclass]
pub struct RBFInterpolator {
    core: RBFCore,
}

#[pymethods]
impl RBFInterpolator {
    #[new]
    #[pyo3(signature = (kernel = "gaussian", epsilon = 1.0))]
    pub fn new(kernel: &str, epsilon: f64) -> PyResult<Self> {
        let kernel = RBFKernel::from_str(kernel).map_err(PyValueError::new_err)?;
        let core = RBFCore::new(kernel, epsilon).map_err(PyValueError::new_err)?;
        Ok(Self { core })
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.core.fit(&x, &y).map_err(PyValueError::new_err)
    }

    pub fn get_weights(&self) -> PyResult<Vec<f64>> {
        let weights = self.core.weights().map_err(PyValueError::new_err)?;
        Ok(weights.to_vec())
    }

    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        py_call_impl(&self.core, py, x)
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
