/// Least Squares Polynomial Approximation module reusing shared core.
use crate::least_squares_core::LeastSquaresCore;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
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
        if let Ok(single) = x.extract::<f64>() {
            let result = self
                .core
                .evaluate_single(single)
                .map_err(PyValueError::new_err)?;
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

        if let Ok(v) = x.extract::<Vec<f64>>() {
            let results = self.core.evaluate_many(&v).map_err(PyValueError::new_err)?;
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
