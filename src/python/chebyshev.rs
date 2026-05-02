/// Python wrapper for the shared Chebyshev core.
use crate::core::chebyshev_core::ChebyshevCore;
use crate::python::pywrap_macros::py_call_impl;
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
        py_call_impl(&self.core, py, x)
    }

    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
