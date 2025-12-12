use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Core Lagrange interpolation function
pub fn lagrange_interpolation(x_values: &[f64], y_values: &[f64], x: f64) -> f64 {
    let n: usize = x_values.len();
    let mut result = 0.0;
    for i in 0..n {
        let mut term: f64 = y_values[i];
        for j in 0..n {
            if i != j {
                term *= (x - x_values[j]) / (x_values[i] - x_values[j]);
            }
        }
        result += term;
    }
    result
}

#[pyclass]
pub struct LagrangeInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl LagrangeInterpolator {
    #[new]
    pub fn new() -> Self {
        LagrangeInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            fitted: false,
        }
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() {
            return Err(PyValueError::new_err(
                "x and y must have the same length"
            ));
        }
        if x.is_empty() {
            return Err(PyValueError::new_err(
                "x and y cannot be empty"
            ));
        }
        
        self.x_values = x;
        self.y_values = y;
        self.fitted = true;
        Ok(())
    }

    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        if let Ok(single_x) = x.extract::<f64>() {
            let result = lagrange_interpolation(&self.x_values, &self.y_values, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| lagrange_interpolation(&self.x_values, &self.y_values, xi))
                .collect();
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats"
        ))
    }

    pub fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "LagrangeInterpolator(fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "LagrangeInterpolator(not fitted)".to_string()
        }
    }
}