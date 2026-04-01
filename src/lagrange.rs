/// Lagrange Interpolation Module (Barycentric Form)
///
/// Thin PyO3 wrapper over the shared `LagrangeCore`.
use crate::lagrange_core::LagrangeCore;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
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
        if let Ok(single) = x.extract::<f64>() {
            let value = self
                .core
                .evaluate_single(single)
                .map_err(PyValueError::new_err)?;
            return Ok(value.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let x_slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
            let result_slice = unsafe { result_array.as_slice_mut()? };
            self.core
                .fill_many(x_slice, result_slice)
                .map_err(PyValueError::new_err)?;
            return Ok(result_array.into_any().unbind());
        }

        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results = self
                .core
                .evaluate_many(&x_list)
                .map_err(PyValueError::new_err)?;
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
