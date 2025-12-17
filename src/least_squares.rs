/// Least Squares Polynomial Approximation Module
/// 
/// This module implements least squares polynomial fitting, which finds the
/// polynomial of specified degree that best fits the data in a least squares sense.
/// Unlike interpolation, it does NOT pass through all points exactly.
/// 
/// # Mathematical Background
/// 
/// For a polynomial P(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ of degree n,
/// least squares minimizes the sum of squared errors:
/// 
/// min Σᵢ (yᵢ - P(xᵢ))²
/// 
/// This leads to the normal equations: (AᵀA)c = Aᵀy
/// where A is the Vandermonde matrix.
/// 
/// # Characteristics
/// 
/// - **Approximation**: Does NOT pass through all points (smooths data)
/// - **Flexible degree**: Choose polynomial degree (typically < n_points/2)
/// - **Noise handling**: Excellent for noisy measurements
/// - **Regularization**: Lower degree provides smoothing
/// - **Complexity**: O(nm²) where n=points, m=degree
/// - **Quality metric**: R² coefficient of determination
/// 
/// # R² Interpretation
/// 
/// R² = 1 - (SS_residual / SS_total)
/// - R² = 1.0: Perfect fit
/// - R² = 0.9: 90% of variance explained
/// - R² < 0.5: Poor fit
/// 
/// # Degree Selection Guidelines
/// 
/// - Too low: Underfitting (high bias)
/// - Too high: Overfitting (high variance, oscillations)
/// - Rule of thumb: degree ≤ n_points / 3
/// - For noisy data: degree = 2-5 often optimal
/// 
/// # Use Cases
/// 
/// - Experimental data with measurement errors
/// - Trend analysis and smoothing
/// - Denoising signals
/// - Physical law fitting (linear, quadratic, etc.)
/// - When exact fit is not desired
/// 
/// # Comparison with Interpolation
/// 
/// | Aspect | Least Squares | Interpolation |
/// |--------|---------------|---------------|
/// | Exact fit | No | Yes |
/// | Noise handling | Excellent | Poor (amplifies) |
/// | Smoothness | Very smooth | Depends on method |
/// | Oscillations | Controlled | Can occur |
/// | Use case | Noisy data | Clean data |
/// 
/// # Examples
/// 
/// ```python
/// from interlib import LeastSquaresInterpolator
/// import numpy as np
/// 
/// # Noisy data
/// x = np.linspace(0, 10, 50)
/// y_clean = 2 + 3*x - 0.5*x**2
/// y_noisy = y_clean + np.random.normal(0, 2, 50)
/// 
/// # Fit quadratic (degree 2)
/// ls = LeastSquaresInterpolator(degree=2)
/// ls.fit(x.tolist(), y_noisy.tolist())
/// 
/// # Get coefficients [c₀, c₁, c₂, ...]
/// coeffs = ls.get_coefficients()
/// print(f"Polynomial: {coeffs[0]:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x²")
/// 
/// # Check fit quality
/// r2 = ls.r_squared()
/// print(f"R² = {r2:.4f}")
/// 
/// # Evaluate
/// y_pred = ls(x.tolist())
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Solve linear system using Gaussian elimination with partial pivoting
/// 
/// Solves Ax = b using forward elimination and back substitution with
/// row pivoting for numerical stability.
/// 
/// # Arguments
/// 
/// * `a` - Coefficient matrix (modified in-place)
/// * `b` - Right-hand side vector (modified in-place)
/// 
/// # Returns
/// 
/// Solution vector x, or error message if matrix is singular
/// 
/// # Algorithm
/// 
/// 1. Forward elimination with partial pivoting
/// 2. Back substitution
/// 3. Complexity: O(n³)
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    
    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot (largest element in column k)
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
        
        // Eliminate column k below diagonal
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
/// 
/// Solves the normal equations (AᵀA)c = Aᵀy where A is the Vandermonde matrix.
/// 
/// # Arguments
/// 
/// * `x_values` - X coordinates of data points
/// * `y_values` - Y coordinates of data points
/// * `degree` - Degree of polynomial to fit
/// 
/// # Returns
/// 
/// Polynomial coefficients [c₀, c₁, c₂, ..., cₙ] or error message
fn least_squares_polynomial(x_values: &[f64], y_values: &[f64], degree: usize) -> Result<Vec<f64>, String> {
    let n = x_values.len();
    let m = degree + 1; // Number of coefficients
    
    if n < m {
        return Err(format!("Need at least {} points for degree {} polynomial", m, degree));
    }
    
    // Build normal equations: (AᵀA)c = Aᵀb
    let mut ata = vec![vec![0.0; m]; m];
    let mut atb = vec![0.0; m];
    
    // Compute AᵀA
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x_values[k].powi((i + j) as i32);
            }
            ata[i][j] = sum;
        }
        
        // Compute Aᵀb
        let mut sum = 0.0;
        for k in 0..n {
            sum += y_values[k] * x_values[k].powi(i as i32);
        }
        atb[i] = sum;
    }
    
    solve_linear_system(ata, atb)
}

/// Evaluate polynomial with given coefficients
/// 
/// Computes P(x) = c₀ + c₁x + c₂x² + ... using Horner's method for stability.
/// 
/// # Arguments
/// 
/// * `coefficients` - Polynomial coefficients [c₀, c₁, ..., cₙ]
/// * `x` - Point at which to evaluate
/// 
/// # Returns
/// 
/// The value of the polynomial at x
fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    let mut x_power = 1.0;
    
    for &coef in coefficients {
        result += coef * x_power;
        x_power *= x;
    }
    
    result
}

/// Least Squares Polynomial Approximator
/// 
/// Fits a polynomial of specified degree that minimizes sum of squared errors.
/// Provides R² metric to assess quality of fit.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of data points
/// * `y_values` - Stored y coordinates of data points
/// * `coefficients` - Computed polynomial coefficients
/// * `degree` - Degree of the polynomial
/// * `fitted` - Whether the approximator has been fitted
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
    /// Create a new least squares approximator
    /// 
    /// Parameters
    /// ----------
    /// degree : int, default=2
    ///     Degree of polynomial to fit
    ///     
    ///     Guidelines:
    ///     - degree=1: Linear regression
    ///     - degree=2-3: Most common for noisy data
    ///     - degree=4-6: For more complex patterns
    ///     - degree > 10: Usually not recommended (overfitting)
    /// 
    /// Returns
    /// -------
    /// LeastSquaresInterpolator
    ///     A new, unfitted approximator instance
    /// 
    /// Examples
    /// --------
    /// >>> ls = LeastSquaresInterpolator(degree=2)  # Quadratic fit
    /// >>> ls = LeastSquaresInterpolator(degree=1)  # Linear fit
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

    /// Fit the polynomial to data points
    /// 
    /// Computes the least squares solution that minimizes Σ(yᵢ - P(xᵢ))².
    /// 
    /// Parameters
    /// ----------
    /// x : list of float
    ///     X coordinates of data points
    /// y : list of float
    ///     Y coordinates of data points
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If x and y have different lengths
    ///     If fewer than degree+1 points are provided
    ///     If fitting fails (singular matrix)
    /// 
    /// Notes
    /// -----
    /// Requires at least degree+1 data points. More points improve stability
    /// and allow the method to smooth noise effectively.
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

    /// Get the polynomial coefficients
    /// 
    /// Returns coefficients in ascending degree order: [c₀, c₁, c₂, ...]
    /// representing the polynomial c₀ + c₁x + c₂x² + ...
    /// 
    /// Returns
    /// -------
    /// list of float
    ///     Polynomial coefficients [constant, linear, quadratic, ...]
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the approximator has not been fitted
    /// 
    /// Examples
    /// --------
    /// >>> ls.fit([0, 1, 2], [1, 3, 4])
    /// >>> coeffs = ls.get_coefficients()
    /// >>> print(f"y = {coeffs[0]:.2f} + {coeffs[1]:.2f}x")
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Get the polynomial degree
    /// 
    /// Returns
    /// -------
    /// int
    ///     Degree of the fitted polynomial
    pub fn get_degree(&self) -> usize {
        self.degree
    }

    /// Compute R-squared (coefficient of determination)
    /// 
    /// R² measures the proportion of variance in y explained by the model.
    /// 
    /// Returns
    /// -------
    /// float
    ///     R² value between 0 and 1
    ///     - 1.0: Perfect fit (all variance explained)
    ///     - 0.9: Very good fit (90% variance explained)
    ///     - 0.5: Moderate fit
    ///     - 0.0: Model no better than mean
    ///     - <0.0: Model worse than mean (rare)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the approximator has not been fitted
    /// 
    /// Notes
    /// -----
    /// R² = 1 - (SS_res / SS_tot) where:
    /// - SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
    /// - SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)
    pub fn r_squared(&self) -> PyResult<f64> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        
        // Calculate mean of y values
        let y_mean: f64 = self.y_values.iter().sum::<f64>() / self.y_values.len() as f64;
        
        // Calculate SS_tot and SS_res
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

    /// Evaluate the polynomial at one or more points
    /// 
    /// Parameters
    /// ----------
    /// x : float or list of float
    ///     Point(s) at which to evaluate
    /// 
    /// Returns
    /// -------
    /// float or list of float
    ///     Approximated value(s) at the specified point(s)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the approximator has not been fitted
    ///     If input is neither a float nor a list of floats
    /// 
    /// Notes
    /// -----
    /// Can safely extrapolate, but extrapolation quality depends on
    /// polynomial degree and data distribution.
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

    /// String representation of the approximator
    /// 
    /// Returns
    /// -------
    /// str
    ///     Description including degree, number of points, and R² if fitted
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