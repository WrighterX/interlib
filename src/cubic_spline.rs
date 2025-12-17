/// Cubic Spline Interpolation Module
/// 
/// This module implements natural cubic spline interpolation, which constructs
/// a piecewise cubic polynomial that is smooth (C² continuous) across all data points.
/// 
/// # Mathematical Background
/// 
/// For n data points, cubic spline creates n-1 cubic polynomial segments:
/// 
/// Sᵢ(x) = aᵢ + bᵢ(x-xᵢ) + cᵢ(x-xᵢ)² + dᵢ(x-xᵢ)³  for x ∈ [xᵢ, xᵢ₊₁]
/// 
/// # Continuity Conditions
/// 
/// - C⁰: Function values match at all points
/// - C¹: First derivatives match at interior points (smooth)
/// - C²: Second derivatives match at interior points (no sudden curvature changes)
/// - Natural boundary: Second derivative = 0 at endpoints
/// 
/// # Characteristics
/// 
/// - **Smooth**: C² continuous (continuous up to second derivative)
/// - **Local control**: Changing one point affects only nearby segments
/// - **No oscillations**: Avoids Runge's phenomenon
/// - **Industry standard**: Used in CAD, graphics, data analysis
/// - **Complexity**: O(n) computation via tridiagonal solver, O(log n) evaluation
/// 
/// # Advantages
/// 
/// - Smooth, visually pleasing curves
/// - Stable numerical properties
/// - Mimics natural flexible objects (drafting splines)
/// - Excellent for differentiable functions
/// 
/// # Use Cases
/// 
/// - Curve fitting in engineering and science
/// - Computer graphics and CAD
/// - Animation and motion planning
/// - Time series interpolation
/// - Any application requiring smooth interpolation
/// 
/// # Examples
/// 
/// ```python
/// from interlib import CubicSplineInterpolator
/// 
/// # Create interpolator
/// spline = CubicSplineInterpolator()
/// 
/// # Fit with data points
/// x = [0.0, 1.0, 2.0, 3.0, 4.0]
/// y = [0.0, 1.0, 4.0, 9.0, 16.0]
/// spline.fit(x, y)
/// 
/// # Get number of segments
/// n_seg = spline.num_segments()
/// 
/// # Evaluate
/// result = spline(2.5)
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Cubic spline segment representation
/// 
/// Represents one cubic polynomial segment: a + b(x-xᵢ) + c(x-xᵢ)² + d(x-xᵢ)³
#[derive(Clone, Debug)]
struct SplineSegment {
    a: f64, // Constant term (function value at left endpoint)
    b: f64, // Linear coefficient (related to first derivative)
    c: f64, // Quadratic coefficient (related to second derivative)
    d: f64, // Cubic coefficient
    x: f64, // Left endpoint of segment
}

impl SplineSegment {
    /// Evaluate the cubic spline segment at a point
    /// 
    /// # Arguments
    /// 
    /// * `x_val` - Point at which to evaluate (should be in segment range)
    /// 
    /// # Returns
    /// 
    /// The value of the cubic polynomial at x_val
    fn eval(&self, x_val: f64) -> f64 {
        let dx = x_val - self.x;
        self.a + self.b * dx + self.c * dx * dx + self.d * dx * dx * dx
    }
}

/// Solve tridiagonal linear system using Thomas algorithm
/// 
/// Efficiently solves a tridiagonal system Ax = d where A has:
/// - Main diagonal: b[i]
/// - Lower diagonal: a[i]
/// - Upper diagonal: c[i]
/// 
/// # Arguments
/// 
/// * `a` - Lower diagonal (not used for first equation)
/// * `b` - Main diagonal
/// * `c` - Upper diagonal (not used for last equation)
/// * `d` - Right-hand side
/// 
/// # Returns
/// 
/// Solution vector x
/// 
/// # Algorithm
/// 
/// Thomas algorithm (specialized Gaussian elimination for tridiagonal systems):
/// - Time complexity: O(n)
/// - Space complexity: O(n)
/// - More efficient than general Gaussian elimination O(n³)
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];
    
    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    
    for i in 1..n {
        let m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
        c_prime[i] = c[i] * m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m;
    }
    
    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    
    x
}

/// Compute natural cubic spline coefficients
/// 
/// Constructs a natural cubic spline through the given data points.
/// Natural boundary conditions: S''(x₀) = S''(xₙ) = 0
/// 
/// # Arguments
/// 
/// * `x_values` - X coordinates of data points (must be sorted)
/// * `y_values` - Y coordinates of data points
/// 
/// # Returns
/// 
/// Vector of SplineSegment structs, one for each interval
fn compute_natural_cubic_spline(x_values: &[f64], y_values: &[f64]) -> Vec<SplineSegment> {
    let n = x_values.len();
    
    if n < 2 {
        return vec![];
    }
    
    // Calculate h values (interval widths)
    let mut h = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = x_values[i + 1] - x_values[i];
    }
    
    // Set up tridiagonal system for second derivatives (c coefficients)
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    
    // First row (natural boundary: c[0] = 0)
    b[0] = 1.0;
    c[0] = 0.0;
    d[0] = 0.0;
    
    // Interior rows (continuity of second derivative)
    for i in 1..n - 1 {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 3.0 * ((y_values[i + 1] - y_values[i]) / h[i] 
                    - (y_values[i] - y_values[i - 1]) / h[i - 1]);
    }
    
    // Last row (natural boundary: c[n-1] = 0)
    a[n - 1] = 0.0;
    b[n - 1] = 1.0;
    d[n - 1] = 0.0;
    
    // Solve for second derivatives
    let c_coeffs = solve_tridiagonal(&a, &b, &c, &d);
    
    // Build spline segments from coefficients
    let mut segments = Vec::new();
    
    for i in 0..n - 1 {
        let segment = SplineSegment {
            a: y_values[i],
            b: (y_values[i + 1] - y_values[i]) / h[i] 
                - h[i] * (c_coeffs[i + 1] + 2.0 * c_coeffs[i]) / 3.0,
            c: c_coeffs[i],
            d: (c_coeffs[i + 1] - c_coeffs[i]) / (3.0 * h[i]),
            x: x_values[i],
        };
        segments.push(segment);
    }
    
    segments
}

/// Evaluate cubic spline at a single point
/// 
/// Finds the appropriate segment and evaluates the cubic polynomial.
/// 
/// # Arguments
/// 
/// * `segments` - Pre-computed spline segments
/// * `x_values` - X coordinates of data points
/// * `x` - Point at which to evaluate
/// 
/// # Returns
/// 
/// The interpolated value at x
fn cubic_spline_eval(segments: &[SplineSegment], x_values: &[f64], x: f64) -> f64 {
    if segments.is_empty() {
        return f64::NAN;
    }
    
    let n = x_values.len();
    
    // Handle boundary cases (extrapolate using edge segments)
    if x <= x_values[0] {
        return segments[0].eval(x);
    }
    if x >= x_values[n - 1] {
        return segments[segments.len() - 1].eval(x);
    }
    
    // Find the segment containing x
    for i in 0..segments.len() {
        if x >= x_values[i] && x <= x_values[i + 1] {
            return segments[i].eval(x);
        }
    }
    
    f64::NAN
}

/// Natural Cubic Spline Interpolator
/// 
/// A stateful interpolator that pre-computes cubic spline segments
/// for smooth C² continuous interpolation.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of data points
/// * `y_values` - Stored y coordinates of data points
/// * `segments` - Pre-computed cubic polynomial segments
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct CubicSplineInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    segments: Vec<SplineSegment>,
    fitted: bool,
}

#[pymethods]
impl CubicSplineInterpolator {
    /// Create a new cubic spline interpolator
    /// 
    /// Returns
    /// -------
    /// CubicSplineInterpolator
    ///     A new, unfitted interpolator instance
    #[new]
    pub fn new() -> Self {
        CubicSplineInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            segments: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points
    /// 
    /// Computes the natural cubic spline segments. Natural boundary conditions
    /// are used: second derivative equals zero at both endpoints.
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
    ///     If fewer than 2 data points are provided
    ///     If x values are not strictly increasing
    /// 
    /// Notes
    /// -----
    /// The spline segments are computed using the Thomas algorithm to solve
    /// the tridiagonal system for second derivatives, which is O(n) efficient.
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() {
            return Err(PyValueError::new_err(
                "x and y must have the same length"
            ));
        }
        if x.len() < 2 {
            return Err(PyValueError::new_err(
                "Cubic spline interpolation requires at least 2 data points"
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
        
        // Compute spline coefficients
        self.segments = compute_natural_cubic_spline(&self.x_values, &self.y_values);
        
        if self.segments.is_empty() {
            return Err(PyValueError::new_err(
                "Failed to compute spline coefficients"
            ));
        }
        
        self.fitted = true;
        Ok(())
    }

    /// Get the number of spline segments
    /// 
    /// Returns
    /// -------
    /// int
    ///     Number of cubic polynomial segments (n_points - 1)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    pub fn num_segments(&self) -> PyResult<usize> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.segments.len())
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
    ///     Interpolated value(s) at the specified point(s)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If input is neither a float nor a list of floats
    /// 
    /// Notes
    /// -----
    /// For points outside the data range, the edge segments are used
    /// for extrapolation (linear extrapolation from the edge cubic).
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = cubic_spline_eval(&self.segments, &self.x_values, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| cubic_spline_eval(&self.segments, &self.x_values, xi))
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
    ///     Description of the interpolator state including number of segments
    pub fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "CubicSplineInterpolator(fitted with {} points, {} segments, x range: [{:.2}, {:.2}])",
                self.x_values.len(),
                self.segments.len(),
                self.x_values.first().unwrap_or(&0.0),
                self.x_values.last().unwrap_or(&0.0)
            )
        } else {
            "CubicSplineInterpolator(not fitted)".to_string()
        }
    }
}