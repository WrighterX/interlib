/// Python wrapper for the shared Chebyshev core.
use crate::chebyshev_core::ChebyshevCore;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
pub struct ChebyshevInterpolator {
    core: ChebyshevCore,
}

#[pymethods]
impl ChebyshevInterpolator {
    #[new]
    #[pyo3(signature = (n_points=10, x_min=-1.0, x_max=1.0, use_clenshaw=true))]
    pub fn new(n_points: usize, x_min: f64, x_max: f64, use_clenshaw: bool) -> PyResult<Self> {
        let core = ChebyshevCore::new(n_points, x_min, x_max, use_clenshaw)
            .map_err(PyValueError::new_err)?;
        Ok(Self { core })
    }

    pub fn get_nodes(&self) -> Vec<f64> {
        self.core.nodes().to_vec()
    }

    pub fn fit(&mut self, y: Vec<f64>) -> PyResult<()> {
        self.core.fit(&y).map_err(PyValueError::new_err)
    }

    pub fn fit_function(&mut self, py: Python<'_>, func: Py<PyAny>) -> PyResult<()> {
        let mut values = Vec::with_capacity(self.core.n_points());
        for &x in self.core.nodes() {
            let result = func.call1(py, (x,))?;
            let value: f64 = result.extract(py)?;
            values.push(value);
        }
        self.core.fit(&values).map_err(PyValueError::new_err)
    }

    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        let coeffs = self.core.coefficients().map_err(PyValueError::new_err)?;
        Ok(coeffs.to_vec())
    }

    pub fn set_method(&mut self, use_clenshaw: bool) {
        self.core.set_method(use_clenshaw);
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
            let mut results = Vec::with_capacity(list.len());
            for value in list {
                let result = self
                    .core
                    .evaluate_single(value)
                    .map_err(PyValueError::new_err)?;
                results.push(result);
            }
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
