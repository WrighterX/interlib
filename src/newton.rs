//! Newton Interpolation module

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Compute divided differences for Newton interpolation
fn divided_differences(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut coef = ys.to_vec();
    
    for j in 1..n {
        for i in (j..n).rev() {
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j]);
        }
    }
    coef
}

/// Evaluate Newton polynomial at a single point
fn newton_evaluate(xs: &[f64], coef: &[f64], x: f64) -> f64 {
    let n = coef.len();
    let mut result = coef[n - 1];
    
    for i in (0..n - 1).rev() {
        result = result * (x - xs[i]) + coef[i];
    }
    result
}

#[pyclass]
pub struct NewtonInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    coefficients: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl NewtonInterpolator {
    #[new]
    pub fn new() -> Self {
        NewtonInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with x and y data points
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
        
        // Compute Newton coefficients during fitting
        self.coefficients = divided_differences(&self.x_values, &self.y_values);
        
        self.fitted = true;
        Ok(())
    }

    /// Get the Newton polynomial coefficients
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Evaluate the interpolation at a single point or multiple points
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = newton_evaluate(&self.x_values, &self.coefficients, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| newton_evaluate(&self.x_values, &self.coefficients, xi))
                .collect();
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats"
        ))
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "NewtonInterpolator(fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "NewtonInterpolator(not fitted)".to_string()
        }
    }
}