use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Solve 3x3 system for quadratic coefficients using Cramer's rule
/// For quadratic: y = a + b*x + c*x^2
/// Given three points (x0,y0), (x1,y1), (x2,y2)
fn solve_quadratic_coefficients(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64) {
    // System of equations:
    // a + b*x0 + c*x0^2 = y0
    // a + b*x1 + c*x1^2 = y1
    // a + b*x2 + c*x2^2 = y2
    
    let x0_sq = x0 * x0;
    let x1_sq = x1 * x1;
    let x2_sq = x2 * x2;
    
    // Calculate determinant of coefficient matrix
    let det = 1.0 * (x1 * x2_sq - x2 * x1_sq)
            - x0 * (1.0 * x2_sq - 1.0 * x1_sq)
            + x0_sq * (1.0 * x2 - 1.0 * x1);
    
    if det.abs() < 1e-10 {
        // Points are colinear or det is too small, fallback to linear
        let slope = (y1 - y0) / (x1 - x0);
        let intercept = y0 - slope * x0;
        return (intercept, slope, 0.0);
    }
    
    // Using Cramer's rule
    let det_a = y0 * (x1 * x2_sq - x2 * x1_sq)
              - y1 * (x0 * x2_sq - x2 * x0_sq)
              + y2 * (x0 * x1_sq - x1 * x0_sq);
    
    let det_b = 1.0 * (y1 * x2_sq - y2 * x1_sq)
              - x0 * (y0 * x2_sq - y2 * x0_sq)
              + x0_sq * (y0 * x2 - y2 * x1);
    
    // Actually, let's use direct formula for c
    let det_c = y0 * (x1 - x2) + y1 * (x2 - x0) + y2 * (x0 - x1);
    
    let a = det_a / det;
    let b = det_b / det;
    let c = det_c / det;
    
    (a, b, c)
}

/// Evaluate quadratic polynomial at x
fn eval_quadratic(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a + b * x + c * x * x
}

/// Perform piecewise quadratic interpolation at a single point
fn quadratic_interpolate_single(x_values: &[f64], y_values: &[f64], x: f64) -> f64 {
    let n = x_values.len();
    
    // Handle edge cases
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return y_values[0];
    }
    if n == 2 {
        // Fall back to linear interpolation
        let t = (x - x_values[0]) / (x_values[1] - x_values[0]);
        return y_values[0] + t * (y_values[1] - y_values[0]);
    }
    
    // Handle boundary cases - use first/last quadratic segment
    if x <= x_values[0] {
        // Use first three points
        let (a, b, c) = solve_quadratic_coefficients(
            x_values[0], y_values[0],
            x_values[1], y_values[1],
            x_values[2], y_values[2]
        );
        return eval_quadratic(a, b, c, x);
    }
    
    if x >= x_values[n - 1] {
        // Use last three points
        let (a, b, c) = solve_quadratic_coefficients(
            x_values[n - 3], y_values[n - 3],
            x_values[n - 2], y_values[n - 2],
            x_values[n - 1], y_values[n - 1]
        );
        return eval_quadratic(a, b, c, x);
    }
    
    // Find the interval containing x
    for i in 0..n - 1 {
        if x >= x_values[i] && x <= x_values[i + 1] {
            // Use three points centered around this interval
            let idx = if i == 0 {
                0 // Use points [0, 1, 2]
            } else if i == n - 2 {
                n - 3 // Use points [n-3, n-2, n-1]
            } else {
                i - 1 // Use points [i-1, i, i+1]
            };
            
            let (a, b, c) = solve_quadratic_coefficients(
                x_values[idx], y_values[idx],
                x_values[idx + 1], y_values[idx + 1],
                x_values[idx + 2], y_values[idx + 2]
            );
            
            return eval_quadratic(a, b, c, x);
        }
    }
    
    f64::NAN
}

#[pyclass]
pub struct QuadraticInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl QuadraticInterpolator {
    #[new]
    pub fn new() -> Self {
        QuadraticInterpolator {
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
        if x.len() < 3 {
            return Err(PyValueError::new_err(
                "Quadratic interpolation requires at least 3 data points"
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
            let result = quadratic_interpolate_single(&self.x_values, &self.y_values, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| quadratic_interpolate_single(&self.x_values, &self.y_values, xi))
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
                "QuadraticInterpolator(fitted with {} points, x range: [{:.2}, {:.2}])",
                self.x_values.len(),
                self.x_values.first().unwrap_or(&0.0),
                self.x_values.last().unwrap_or(&0.0)
            )
        } else {
            "QuadraticInterpolator(not fitted)".to_string()
        }
    }
}