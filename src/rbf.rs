/// Radial Basis Function (RBF) interpolation for Python plus FFI consumers.
///
/// This module exposes the Python-facing `RBFInterpolator`, which wraps the
/// shared `RBFCore` implementation to keep the interpolation logic consistent
/// between the Python bindings and the MATLAB/FFI layers. Supported kernels are
/// Gaussian, multiquadric, inverse multiquadric, thin plate spline, and linear.
use crate::rbf_core::{RBFCore, RBFKernel};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

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
        if let Ok(single) = x.extract::<f64>() {
            let value = self.core.evaluate_single(single).map_err(PyValueError::new_err)?;
            return Ok(value.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [slice.len()], false) };
            {
                let result_slice = unsafe { result_array.as_slice_mut()? };
                self.core
                    .fill_many(slice, result_slice)
                    .map_err(PyValueError::new_err)?;
            }
            return Ok(result_array.into_any().unbind());
        }

        if let Ok(list) = x.extract::<Vec<f64>>() {
            let values = self
                .core
                .evaluate_many(&list)
                .map_err(PyValueError::new_err)?;
            return Ok(values.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, list of floats, or NumPy array",
        ))
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
