use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Perform linear interpolation at a single point
fn linear_interpolate_single(x_values: &[f64], y_values: &[f64], x: f64) -> f64 {
    let n = x_values.len();
    
    // Handle edge cases
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return y_values[0];
    }
    
    // Check if x is outside the range
    if x <= x_values[0] {
        return y_values[0];
    }
    if x >= x_values[n - 1] {
        return y_values[n - 1];
    }
    
    // Find the interval [x0, x1] containing x
    for i in 0..n - 1 {
        let x0 = x_values[i];
        let x1 = x_values[i + 1];
        
        if x >= x0 && x <= x1 {
            let y0 = y_values[i];
            let y1 = y_values[i + 1];
            
            // Linear interpolation formula
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    
    // Should not reach here, but return NaN as fallback
    f64::NAN
}

#[pyclass]
pub struct LinearInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl LinearInterpolator {
    #[new]
    pub fn new() -> Self {
        LinearInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
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
        
        // Check if x values are sorted
        for i in 0..x.len() - 1 {
            if x[i] >= x[i + 1] {
                return Err(PyValueError::new_err(
                    "x values must be strictly increasing"
                ));
            }
        }
        
        self.x_values = x;
        self.y_values = y;
        self.fitted = true;
        Ok(())
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
            let result = linear_interpolate_single(&self.x_values, &self.y_values, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| linear_interpolate_single(&self.x_values, &self.y_values, xi))
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
                "LinearInterpolator(fitted with {} points, x range: [{:.2}, {:.2}])",
                self.x_values.len(),
                self.x_values.first().unwrap_or(&0.0),
                self.x_values.last().unwrap_or(&0.0)
            )
        } else {
            "LinearInterpolator(not fitted)".to_string()
        }
    }
}