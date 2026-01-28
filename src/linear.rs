/// Linear Interpolation Module
/// 
/// This module implements piecewise linear interpolation, which connects
/// consecutive data points with straight line segments.
/// 
/// # Mathematical Background
/// 
/// For x in [xᵢ, xᵢ₊₁], the linear interpolation formula is:
/// 
/// y(x) = yᵢ + (yᵢ₊₁ - yᵢ) * (x - xᵢ) / (xᵢ₊₁ - xᵢ)
/// 
/// Or equivalently using parameter t ∈ [0, 1]:
/// 
/// y(x) = (1 - t) * yᵢ + t * yᵢ₊₁
/// where t = (x - xᵢ) / (xᵢ₊₁ - xᵢ)
/// 
/// # Characteristics
/// 
/// - **Simplest method**: Easy to understand and implement
/// - **C⁰ continuous**: Continuous but not smooth (corners at data points)
/// - **Fast**: O(n) linear search or O(log n) with binary search
/// - **No oscillations**: Monotone between points if data is monotone
/// - **Local**: Each segment is independent
/// - **Memory efficient**: No coefficient computation needed
/// - **Exact at points**: Passes through all data points
/// 
/// # Advantages
/// 
/// - Very fast computation
/// - Minimal memory usage
/// - Numerically stable
/// - Intuitive behavior
/// - No overshooting or oscillations
/// - Works well for sparse data
/// 
/// # Limitations
/// 
/// - Not differentiable at data points (C⁰ only, not C¹)
/// - Visible corners in the curve
/// - Poor for smooth functions
/// - First derivative is discontinuous
/// 
/// # Use Cases
/// 
/// - Quick data visualization
/// - Real-time applications (fast computation)
/// - Data with natural discontinuities
/// - First approximation or baseline
/// - Lookup tables
/// - When simplicity is priority
/// - Gaming and graphics (LOD systems)
/// 
/// # When NOT to Use
/// 
/// - Smooth curves required
/// - Derivatives needed
/// - High accuracy on smooth functions
/// - Scientific visualization
/// 
/// # Examples
/// 
/// ```python
/// from interlib import LinearInterpolator
/// 
/// # Create interpolator
/// interp = LinearInterpolator()
/// 
/// # Fit with data points
/// x = [0.0, 1.0, 2.0, 3.0]
/// y = [0.0, 2.0, 1.0, 3.0]
/// interp.fit(x, y)
/// 
/// # Evaluate at new points
/// result = interp(1.5)  # = 1.5 (midpoint between 2.0 and 1.0)
/// results = interp([0.5, 1.5, 2.5])
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Perform linear interpolation at a single point
/// 
/// Finds the appropriate interval and computes the linear interpolation.
/// For points outside the data range, uses the edge values (constant extrapolation).
/// 
/// # Arguments
/// 
/// * `x_values` - Array of x coordinates (must be sorted)
/// * `y_values` - Array of y coordinates
/// * `x` - Point at which to evaluate
/// 
/// # Returns
/// 
/// The linearly interpolated value at x
/// 
/// # Algorithm
/// 
/// 1. Handle edge cases (single point, boundaries)
/// 2. Find interval [xᵢ, xᵢ₊₁] containing x (linear search)
/// 3. Compute interpolation parameter t
/// 4. Return yᵢ + t * (yᵢ₊₁ - yᵢ)
fn linear_interpolate_single(x_values: &[f64], y_values: &[f64], x: f64) -> f64 {
    let n = x_values.len();
    
    // Handle edge cases
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return y_values[0];
    }
    
    // Boundary handling: constant extrapolation
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

/// Linear Interpolator
/// 
/// A stateful interpolator that performs piecewise linear interpolation
/// through data points.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of data points
/// * `y_values` - Stored y coordinates of data points
/// * `fitted` - Whether the interpolator has been fitted with data
#[pyclass]
pub struct LinearInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl LinearInterpolator {
    /// Create a new linear interpolator
    /// 
    /// Returns
    /// -------
    /// LinearInterpolator
    ///     A new, unfitted interpolator instance
    /// 
    /// Examples
    /// --------
    /// >>> interp = LinearInterpolator()
    #[new]
    pub fn new() -> Self {
        LinearInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points
    /// 
    /// Stores the data points for later evaluation. No pre-computation is needed
    /// for linear interpolation.
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
    ///     If x or y is empty
    ///     If x values are not strictly increasing
    /// 
    /// Notes
    /// -----
    /// X values must be sorted in strictly increasing order. This is verified
    /// during fitting to ensure correct interpolation behavior.
    /// 
    /// Examples
    /// --------
    /// >>> interp = LinearInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
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
        
        // Check if x values are strictly increasing (TODO: implement auto x sorting?)
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
    ///     Linearly interpolated value(s) at the specified point(s)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If input is neither a float nor a list of floats
    /// 
    /// Notes
    /// -----
    /// For points outside the data range, the interpolator returns the
    /// nearest boundary value (constant extrapolation).
    /// 
    /// Examples
    /// --------
    /// >>> interp = LinearInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp(0.5)  # Midpoint between 0 and 1
    /// 0.5
    /// >>> interp([0.5, 1.5])  # Multiple points
    /// [0.5, 2.5]
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

    /// String representation of the interpolator
    /// 
    /// Returns
    /// -------
    /// str
    ///     Description of the interpolator state
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