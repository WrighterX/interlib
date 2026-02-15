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
fn linear_interpolate_single(x_values: &[f64], y_values: &[f64], slopes: &[f64], x: f64) -> f64 {
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

    // Binary search for the interval containing x
    let idx = match x_values.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(i) => if i >= n - 1 { i - 1 } else { i },
        Err(i) => if i > 0 { i - 1 } else { 0 },
    };

    // Single multiply-add using precomputed slope
    y_values[idx] + slopes[idx] * (x - x_values[idx])
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
    slopes: Vec<f64>,
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
            slopes: Vec::new(),
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
        
        self.slopes = (0..x.len() - 1)
            .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            .collect();
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
            let result = linear_interpolate_single(&self.x_values, &self.y_values, &self.slopes, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| linear_interpolate_single(&self.x_values, &self.y_values, &self.slopes, xi))
                .collect();
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats"
        ))
    }

    /// Replace the data values and recompute slopes
    ///
    /// Because slopes depend on both x and y values, this recomputes
    /// all slopes in O(n). However, the x values (and their sorted order)
    /// are preserved.
    ///
    /// Parameters
    /// ----------
    /// y : list of float
    ///     New data values. Must have the same length as the original x.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted, or if the length
    ///     of y does not match the number of points
    ///
    /// Examples
    /// --------
    /// >>> interp = LinearInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp(0.5)
    /// 0.5
    /// >>> interp.update_y([0.0, 2.0, 6.0])
    /// >>> interp(0.5)
    /// 1.0
    pub fn update_y(&mut self, y: Vec<f64>) -> PyResult<()> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        if y.len() != self.x_values.len() {
            return Err(PyValueError::new_err(
                format!(
                    "y must have length {} (same as x), got {}",
                    self.x_values.len(),
                    y.len()
                )
            ));
        }
        self.slopes = (0..self.x_values.len() - 1)
            .map(|i| (y[i + 1] - y[i]) / (self.x_values[i + 1] - self.x_values[i]))
            .collect();
        self.y_values = y;
        Ok(())
    }

    /// Add a new data point, inserting at the correct sorted position
    ///
    /// Uses binary search to find the insertion index, then recomputes
    /// only the 1–2 affected slopes.
    ///
    /// Parameters
    /// ----------
    /// x_new : float
    ///     The new x coordinate. Must be distinct from all existing x values.
    /// y_new : float
    ///     The data value at the new point
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If x_new duplicates an existing x value
    ///
    /// Examples
    /// --------
    /// >>> interp = LinearInterpolator()
    /// >>> interp.fit([0.0, 2.0, 4.0], [0.0, 4.0, 8.0])
    /// >>> interp.add_point(1.0, 1.0)
    /// >>> interp(0.5)
    /// 0.5
    pub fn add_point(&mut self, x_new: f64, y_new: f64) -> PyResult<()> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Find insertion index via binary search
        let idx = match self.x_values.binary_search_by(|v| v.partial_cmp(&x_new).unwrap()) {
            Ok(_) => {
                return Err(PyValueError::new_err(
                    format!("x_new = {} already exists in the data", x_new)
                ));
            }
            Err(i) => i,
        };

        self.x_values.insert(idx, x_new);
        self.y_values.insert(idx, y_new);

        // Recompute affected slopes
        let n = self.x_values.len(); // new length after insert
        if n < 2 {
            self.slopes.clear();
            return Ok(());
        }

        // The new point at index `idx` affects slopes at idx-1 and idx.
        // The old slope at idx-1 (if it existed) is replaced by two new slopes.
        if idx == 0 {
            // Inserted at the beginning: add a new slope at index 0
            let new_slope = (self.y_values[1] - self.y_values[0])
                / (self.x_values[1] - self.x_values[0]);
            self.slopes.insert(0, new_slope);
            // Also recompute slope at index 1 if it existed (previously was slope 0)
            // Actually the old slope[0] is now at position 1, but it connected old[0]..old[1]
            // which is now new[1]..new[2], so the value is unchanged. No action needed.
        } else if idx == n - 1 {
            // Inserted at the end: add a new slope at the end
            let new_slope = (self.y_values[idx] - self.y_values[idx - 1])
                / (self.x_values[idx] - self.x_values[idx - 1]);
            self.slopes.push(new_slope);
        } else {
            // Inserted in the middle: replace old slope at idx-1 with two new slopes
            let slope_left = (self.y_values[idx] - self.y_values[idx - 1])
                / (self.x_values[idx] - self.x_values[idx - 1]);
            let slope_right = (self.y_values[idx + 1] - self.y_values[idx])
                / (self.x_values[idx + 1] - self.x_values[idx]);
            // Old slope at position idx-1 connected old[idx-1]..old[idx] (now new[idx-1]..new[idx])
            // Replace it with slope_left, then insert slope_right after it
            self.slopes[idx - 1] = slope_left;
            self.slopes.insert(idx, slope_right);
        }

        Ok(())
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