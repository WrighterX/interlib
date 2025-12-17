/// Piecewise Quadratic Interpolation Module
/// 
/// This module implements piecewise quadratic interpolation using overlapping
/// triplets of consecutive data points to fit parabolas.
/// 
/// # Mathematical Background
/// 
/// For each interval, uses three consecutive points to fit a quadratic:
/// 
/// P(x) = a + b(x - x₀) + c(x - x₀)²
/// 
/// The coefficients are found by solving the 3×3 system:
/// 
/// | 1  0   0   | | a |   | y₀ |
/// | 1  h₁  h₁² | | b | = | y₁ |
/// | 1  h₂  h₂² | | c |   | y₂ |
/// 
/// where h₁ = x₁ - x₀, h₂ = x₂ - x₀
/// 
/// # Characteristics
/// 
/// - **Piecewise approach**: Each interval uses local triplet
/// - **C⁰ continuous**: Continuous but not necessarily smooth
/// - **Captures curvature**: Better than linear for curved data
/// - **Local control**: Uses only 3 nearby points per interval
/// - **Complexity**: O(1) per evaluation after interval found
/// 
/// # Advantages over Linear
/// 
/// - Captures quadratic behavior (parabolas)
/// - Smoother appearance than piecewise linear
/// - Better accuracy for curved data
/// - Still computationally efficient
/// 
/// # Advantages over Cubic Spline
/// 
/// - Simpler to understand and implement
/// - Faster evaluation
/// - Less memory (no coefficient table)
/// - Good middle ground
/// 
/// # Limitations
/// 
/// - Not C¹ continuous (derivatives may be discontinuous)
/// - Requires at least 3 points
/// - Less smooth than cubic spline
/// - Can still oscillate slightly
/// 
/// # Use Cases
/// 
/// - Moderately curved data
/// - When cubic spline is overkill
/// - Engineering data with some curvature
/// - Balance between accuracy and simplicity
/// - Trajectory interpolation
/// 
/// # When NOT to Use
/// 
/// - Need C¹ or C² continuity
/// - Very smooth curves required
/// - Data is nearly linear (use linear instead)
/// - High-accuracy scientific work (use cubic spline)
/// 
/// # Examples
/// 
/// ```python
/// from interlib import QuadraticInterpolator
/// 
/// # Create interpolator
/// interp = QuadraticInterpolator()
/// 
/// # Fit with data points (need at least 3)
/// x = [0.0, 1.0, 2.0, 3.0, 4.0]
/// y = [0.0, 1.0, 4.0, 9.0, 16.0]  # y = x²
/// interp.fit(x, y)
/// 
/// # Evaluate - should be exact for quadratic data
/// result = interp(2.5)  # Should be close to 6.25
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Solve 3×3 system for quadratic coefficients using Cramer's rule
/// 
/// Given three points (x₀, y₀), (x₁, y₁), (x₂, y₂), computes coefficients
/// for the quadratic: y = a + b*x + c*x²
/// 
/// # Arguments
/// 
/// * `x0`, `y0` - First point
/// * `x1`, `y1` - Second point
/// * `x2`, `y2` - Third point
/// 
/// # Returns
/// 
/// Tuple (a, b, c) of quadratic coefficients
/// 
/// # Notes
/// 
/// If the points are collinear (determinant near zero), falls back to
/// linear interpolation (c = 0).
fn solve_quadratic_coefficients(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64) {
    // System of equations for quadratic a + b*x + c*x²:
    // a + b*x0 + c*x0² = y0
    // a + b*x1 + c*x1² = y1
    // a + b*x2 + c*x2² = y2
    
    let x0_sq = x0 * x0;
    let x1_sq = x1 * x1;
    let x2_sq = x2 * x2;
    
    // Calculate determinant of coefficient matrix
    let det = 1.0 * (x1 * x2_sq - x2 * x1_sq)
            - x0 * (1.0 * x2_sq - 1.0 * x1_sq)
            + x0_sq * (1.0 * x2 - 1.0 * x1);
    
    // If determinant too small, points are nearly collinear - use linear
    if det.abs() < 1e-10 {
        let slope = (y1 - y0) / (x1 - x0);
        let intercept = y0 - slope * x0;
        return (intercept, slope, 0.0);
    }
    
    // Compute coefficients using Cramer's rule
    let det_a = y0 * (x1 * x2_sq - x2 * x1_sq)
              - y1 * (x0 * x2_sq - x2 * x0_sq)
              + y2 * (x0 * x1_sq - x1 * x0_sq);
    
    let det_b = 1.0 * (y1 * x2_sq - y2 * x1_sq)
              - x0 * (y0 * x2_sq - y2 * x0_sq)
              + x0_sq * (y0 * x2 - y2 * x1);
    
    let det_c = y0 * (x1 - x2) + y1 * (x2 - x0) + y2 * (x0 - x1);
    
    let a = det_a / det;
    let b = det_b / det;
    let c = det_c / det;
    
    (a, b, c)
}

/// Evaluate quadratic polynomial at a point
/// 
/// Computes a + b*x + c*x² efficiently.
/// 
/// # Arguments
/// 
/// * `a`, `b`, `c` - Polynomial coefficients
/// * `x` - Evaluation point
/// 
/// # Returns
/// 
/// Value of the quadratic at x
fn eval_quadratic(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a + b * x + c * x * x
}

/// Perform piecewise quadratic interpolation at a single point
/// 
/// Finds the appropriate triplet of points and evaluates the fitted quadratic.
/// 
/// # Arguments
/// 
/// * `x_values` - Array of x coordinates (sorted)
/// * `y_values` - Array of y coordinates
/// * `x` - Point at which to evaluate
/// 
/// # Returns
/// 
/// The quadratically interpolated value at x
/// 
/// # Algorithm
/// 
/// 1. Find interval [xᵢ, xᵢ₊₁] containing x
/// 2. Select appropriate triplet of points
/// 3. Fit quadratic through the triplet
/// 4. Evaluate at x
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
            // Select triplet centered around this interval
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

/// Piecewise Quadratic Interpolator
/// 
/// A stateful interpolator that performs piecewise quadratic interpolation
/// using overlapping triplets of data points.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of data points
/// * `y_values` - Stored y coordinates of data points
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct QuadraticInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl QuadraticInterpolator {
    /// Create a new quadratic interpolator
    /// 
    /// Returns
    /// -------
    /// QuadraticInterpolator
    ///     A new, unfitted interpolator instance
    #[new]
    pub fn new() -> Self {
        QuadraticInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points
    /// 
    /// Stores the data points. Quadratic segments are computed on-demand
    /// during evaluation.
    /// 
    /// Parameters
    /// ----------
    /// x : list of float
    ///     X coordinates of data points (must be strictly increasing)
    /// y : list of float
    ///     Y coordinates of data points
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If x and y have different lengths
    ///     If fewer than 3 data points are provided
    ///     If x values are not strictly increasing
    /// 
    /// Notes
    /// -----
    /// Requires at least 3 data points to fit quadratics. For 2 points,
    /// falls back to linear interpolation.
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
        
        // Check if x values are strictly increasing
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

    /// Evaluate the interpolation at one or more points
    /// 
    /// Parameters
    /// ----------
    /// x : float or list of float
    ///     Point(s) at which to evaluate the interpolation
    /// 
    /// Returns
    /// -------
    /// float or list of float
    ///     Quadratically interpolated value(s) at the specified point(s)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If input is neither a float nor a list of floats
    /// 
    /// Notes
    /// -----
    /// For each evaluation point, selects the nearest triplet of data points
    /// and fits a quadratic through them.
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

    /// String representation of the interpolator
    /// 
    /// Returns
    /// -------
    /// str
    ///     Description of the interpolator state
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