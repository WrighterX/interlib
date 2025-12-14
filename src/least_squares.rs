use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Solve linear system using Gaussian elimination with partial pivoting
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    
    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();
        
        for i in k + 1..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_idx = i;
            }
        }
        
        // Swap rows if needed
        if max_idx != k {
            a.swap(k, max_idx);
            b.swap(k, max_idx);
        }
        
        // Check for singular matrix
        if a[k][k].abs() < 1e-12 {
            return Err("Matrix is singular or nearly singular".to_string());
        }
        
        // Eliminate column
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    
    Ok(x)
}

/// Compute least squares polynomial coefficients
fn least_squares_polynomial(x_values: &[f64], y_values: &[f64], degree: usize) -> Result<Vec<f64>, String> {
    let n = x_values.len();
    let m = degree + 1; // Number of coefficients
    
    if n < m {
        return Err(format!("Need at least {} points for degree {} polynomial", m, degree));
    }
    
    // Build normal equations: A^T * A * c = A^T * b
    // where A is the Vandermonde matrix
    
    let mut ata = vec![vec![0.0; m]; m];
    let mut atb = vec![0.0; m];
    
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x_values[k].powi((i + j) as i32);
            }
            ata[i][j] = sum;
        }
        
        let mut sum = 0.0;
        for k in 0..n {
            sum += y_values[k] * x_values[k].powi(i as i32);
        }
        atb[i] = sum;
    }
    
    solve_linear_system(ata, atb)
}

/// Evaluate polynomial with given coefficients
fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    let mut x_power = 1.0;
    
    for &coef in coefficients {
        result += coef * x_power;
        x_power *= x;
    }
    
    result
}

#[pyclass]
pub struct LeastSquaresInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    coefficients: Vec<f64>,
    degree: usize,
    fitted: bool,
}

#[pymethods]
impl LeastSquaresInterpolator {
    #[new]
    #[pyo3(signature = (degree=2))]
    pub fn new(degree: usize) -> Self {
        LeastSquaresInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            coefficients: Vec::new(),
            degree,
            fitted: false,
        }
    }

    /// Fit the polynomial approximation with x and y data points
    /// 
    /// Parameters:
    /// -----------
    /// x : list of float
    ///     x coordinates of data points
    /// y : list of float
    ///     y coordinates (function values) at data points
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
        if x.len() <= self.degree {
            return Err(PyValueError::new_err(
                format!("Need at least {} data points for degree {} polynomial", 
                        self.degree + 1, self.degree)
            ));
        }
        
        self.x_values = x;
        self.y_values = y;
        
        // Compute least squares coefficients
        match least_squares_polynomial(&self.x_values, &self.y_values, self.degree) {
            Ok(coef) => {
                self.coefficients = coef;
                self.fitted = true;
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!("Failed to fit: {}", e))),
        }
    }

    /// Get the polynomial coefficients [c0, c1, c2, ...] for c0 + c1*x + c2*x^2 + ...
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Get the degree of the polynomial
    pub fn get_degree(&self) -> usize {
        self.degree
    }

    /// Compute R-squared (coefficient of determination)
    pub fn r_squared(&self) -> PyResult<f64> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        
        // Calculate mean of y values
        let y_mean: f64 = self.y_values.iter().sum::<f64>() / self.y_values.len() as f64;
        
        // Calculate total sum of squares and residual sum of squares
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        
        for i in 0..self.x_values.len() {
            let y_pred = evaluate_polynomial(&self.coefficients, self.x_values[i]);
            ss_res += (self.y_values[i] - y_pred).powi(2);
            ss_tot += (self.y_values[i] - y_mean).powi(2);
        }
        
        if ss_tot == 0.0 {
            return Ok(1.0);
        }
        
        Ok(1.0 - ss_res / ss_tot)
    }

    /// Evaluate the approximation at a single point or multiple points
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = evaluate_polynomial(&self.coefficients, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| evaluate_polynomial(&self.coefficients, xi))
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
                "LeastSquaresInterpolator(degree={}, fitted with {} points, R²={:.4})",
                self.degree,
                self.x_values.len(),
                self.r_squared().unwrap_or(0.0)
            )
        } else {
            format!("LeastSquaresInterpolator(degree={}, not fitted)", self.degree)
        }
    }
}