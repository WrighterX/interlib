/// Hermite Interpolation Module
/// 
/// This module implements Hermite polynomial interpolation, which constructs
/// a polynomial matching both function values AND derivative values at data points.
/// 
/// # Mathematical Background
/// 
/// For n points with derivatives, Hermite interpolation constructs a polynomial
/// of degree at most 2n-1 that satisfies:
/// 
/// - P(xᵢ) = yᵢ  (function value matches)
/// - P'(xᵢ) = y'ᵢ (derivative matches)
/// 
/// This is computed using divided differences with "doubled" points:
/// 
/// Points: x₀, x₀, x₁, x₁, x₂, x₂, ...
/// Values: y₀, y₀, y₁, y₁, y₂, y₂, ...
/// 
/// Where the first divided difference for doubled points uses the derivative:
/// f[xᵢ, xᵢ] = y'ᵢ
/// 
/// # Characteristics
/// 
/// - **Extra constraints**: Uses both values and derivatives (2n conditions)
/// - **Higher degree**: Polynomial of degree 2n-1 for n points
/// - **C¹ continuous**: Smooth first derivative everywhere
/// - **More accurate**: Additional information improves accuracy
/// - **Exact for polynomials**: Exact for polynomials up to degree 2n-1
/// - **Complexity**: O(n²) for coefficient computation, O(n) evaluation
/// 
/// # Advantages
/// 
/// - Much more accurate than standard interpolation
/// - Smooth curve (C¹ continuous)
/// - Useful when derivatives are known or measurable
/// - No need to estimate derivatives from finite differences
/// - Better extrapolation behavior
/// 
/// # Requirements
/// 
/// - Must know or be able to compute derivatives at all points
/// - Derivatives must be reasonably accurate
/// - More computational cost than simple interpolation
/// 
/// # Use Cases
/// 
/// - **Physics**: Position and velocity both known (kinematics)
/// - **Engineering**: Measurement of quantity and its rate of change
/// - **Animation**: Keyframe interpolation with tangent constraints
/// - **CAD/Graphics**: Curve design with slope control
/// - **Differential equations**: Solution interpolation with derivative info
/// - **Trajectory planning**: Position and velocity constraints
/// 
/// # When NOT to Use
/// 
/// - Derivatives unknown or hard to compute
/// - Noisy derivative estimates
/// - When simple interpolation sufficient
/// - Very large datasets (high computational cost)
/// 
/// # Comparison with Other Methods
/// 
/// | Method | Degree | Derivatives | Smoothness |
/// |--------|--------|-------------|------------|
/// | Lagrange | n-1 | Not used | C^∞ |
/// | Hermite | 2n-1 | Required | C¹ |
/// | Cubic Spline | 3(n-1) | Computed | C² |
/// 
/// # Examples
/// 
/// ```python
/// from interlib import HermiteInterpolator
/// import math
/// 
/// # Create interpolator
/// interp = HermiteInterpolator()
/// 
/// # Data: y = sin(x)
/// x = [0.0, 1.0, 2.0, 3.0]
/// y = [math.sin(xi) for xi in x]
/// dy = [math.cos(xi) for xi in x]  # Derivatives
/// 
/// # Fit with both values and derivatives
/// interp.fit(x, y, dy)
/// 
/// # Evaluate
/// result = interp(1.5)  # Very accurate for smooth functions
/// 
/// # Get coefficients
/// coeffs = interp.get_coefficients()
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};

/// Compute Hermite divided differences
///
/// Builds divided differences table using both function values and derivatives.
/// The table is constructed with "doubled" points where consecutive entries
/// for the same x value use the derivative for the first divided difference.
///
/// # Arguments
///
/// * `x_values` - Array of x coordinates
/// * `y_values` - Array of y coordinates (function values)
/// * `dy_values` - Array of derivatives at x coordinates
///
/// # Returns
///
/// Tuple of (z, coefficients) where:
/// - z: Doubled x values [x₀, x₀, x₁, x₁, ...]
/// - coefficients: Divided difference coefficients
///
/// # Algorithm
///
/// 1. Create doubled point array
/// 2. Initialize with function values
/// 3. Use derivatives for f[xᵢ, xᵢ]
/// 4. Compute higher order divided differences
/// 5. Extract diagonal coefficients
fn hermite_divided_differences(
    x_values: &[f64],
    y_values: &[f64],
    dy_values: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = x_values.len();
    let m = 2 * n;  // Total number of points (doubled)

    // OPTIMIZATION 1: Use flat array instead of 2D Vec for better cache locality
    // This eliminates 2n allocations and improves memory access patterns
    let mut z = Vec::with_capacity(m);
    let mut q = vec![0.0; m * m];  // Single allocation: flat row-major array

    // OPTIMIZATION: Precompute stride for flat array access
    // Access q[i][j] as q[i * m + j]
    let stride = m;

    // Initialize z and q[i][0] with doubled points
    for i in 0..n {
        z.push(x_values[i]);
        z.push(x_values[i]);
        q[2 * i * stride] = y_values[i];
        q[(2 * i + 1) * stride] = y_values[i];
    }

    // First divided differences
    for i in 0..n {
        // For doubled points, use derivative
        q[(2 * i + 1) * stride + 1] = dy_values[i];

        // For different points, use standard divided difference
        if i > 0 {
            let idx_cur = 2 * i * stride;
            let idx_prev = (2 * i - 1) * stride;
            let z_cur = z[2 * i];
            let z_prev = z[2 * i - 1];
            q[2 * i * stride + 1] = (q[idx_cur] - q[idx_prev]) / (z_cur - z_prev);
        }
    }

    // OPTIMIZATION 2 & 3: Higher order divided differences with cached lookups
    // Cache z lookups and q accesses for better performance
    // OPTIMIZATION 4: 2-way loop unrolling for SIMD auto-vectorization
    // Process 2 elements per iteration (optimal balance - 4-way had register pressure)
    for j in 2..m {
        let mut i = j;

        // 2-way unrolled loop: process two elements per iteration
        // This allows compiler to auto-vectorize with SIMD instructions (AVX2)
        while i + 1 < m {
            // First element (i)
            let z_i_0 = z[i];
            let z_ij_0 = z[i - j];
            let denominator_0 = z_i_0 - z_ij_0;
            let q_cur_prev_0 = q[i * stride + j - 1];
            let q_prev_row_0 = q[(i - 1) * stride + j - 1];
            q[i * stride + j] = (q_cur_prev_0 - q_prev_row_0) / denominator_0;

            // Second element (i+1) - independent from first, SIMD-friendly
            let z_i_1 = z[i + 1];
            let z_ij_1 = z[i + 1 - j];
            let denominator_1 = z_i_1 - z_ij_1;
            let q_cur_prev_1 = q[(i + 1) * stride + j - 1];
            let q_prev_row_1 = q[i * stride + j - 1];
            q[(i + 1) * stride + j] = (q_cur_prev_1 - q_prev_row_1) / denominator_1;

            i += 2;
        }

        // Handle remainder if m - j is odd
        if i < m {
            let z_i = z[i];
            let z_ij = z[i - j];
            let denominator = z_i - z_ij;
            let q_cur_prev = q[i * stride + j - 1];
            let q_prev_row = q[(i - 1) * stride + j - 1];
            q[i * stride + j] = (q_cur_prev - q_prev_row) / denominator;
        }
    }

    // Extract coefficients (diagonal of q) from flat array
    let mut coefficients = Vec::with_capacity(m);
    for i in 0..m {
        coefficients.push(q[i * stride + i]);
    }

    (z, coefficients)
}

/// Evaluate Hermite polynomial using Horner's method
///
/// Efficiently evaluates the Newton form of the Hermite polynomial.
/// Marked #[inline] to eliminate function call overhead and allow
/// compiler to inline and optimize for each call site.
///
/// # Arguments
///
/// * `z` - Doubled x values array
/// * `coefficients` - Divided difference coefficients
/// * `x` - Point at which to evaluate
///
/// # Returns
///
/// The interpolated value at x
#[inline]
fn hermite_evaluate(z: &[f64], coefficients: &[f64], x: f64) -> f64 {
    let n = coefficients.len();
    if n == 0 {
        return f64::NAN;
    }

    // Horner's method for nested polynomial evaluation
    let mut result = coefficients[n - 1];
    for i in (0..n - 1).rev() {
        result = result * (x - z[i]) + coefficients[i];
    }
    result
}

/// Hermite Polynomial Interpolator
/// 
/// A stateful interpolator that uses both function values and derivatives
/// to construct a smooth polynomial interpolation.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of data points
/// * `y_values` - Stored y coordinates (function values)
/// * `dy_values` - Stored derivatives at data points
/// * `z_values` - Doubled x values for divided differences
/// * `coefficients` - Pre-computed Hermite coefficients
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct HermiteInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    dy_values: Vec<f64>,
    z_values: Vec<f64>,
    coefficients: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl HermiteInterpolator {
    /// Create a new Hermite interpolator
    /// 
    /// Returns
    /// -------
    /// HermiteInterpolator
    ///     A new, unfitted interpolator instance
    #[new]
    pub fn new() -> Self {
        HermiteInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            dy_values: Vec::new(),
            z_values: Vec::new(),
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points and derivatives
    /// 
    /// Computes the Hermite polynomial coefficients using divided differences
    /// with doubled points.
    /// 
    /// Parameters
    /// ----------
    /// x : list of float
    ///     X coordinates of data points
    /// y : list of float
    ///     Y coordinates (function values) at data points
    /// dy : list of float
    ///     Derivatives (dy/dx) at data points
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If x, y, and dy don't all have the same length
    ///     If any of the arrays is empty
    /// 
    /// Notes
    /// -----
    /// The quality of interpolation depends on the accuracy of the derivative
    /// values. Inaccurate derivatives can lead to poor interpolation results.
    /// 
    /// Examples
    /// --------
    /// >>> import math
    /// >>> interp = HermiteInterpolator()
    /// >>> x = [0.0, 1.0, 2.0]
    /// >>> y = [0.0, 1.0, 8.0]  # x³
    /// >>> dy = [0.0, 3.0, 12.0]  # 3x²
    /// >>> interp.fit(x, y, dy)
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>, dy: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() || x.len() != dy.len() {
            return Err(PyValueError::new_err(
                "x, y, and dy must all have the same length"
            ));
        }
        if x.is_empty() {
            return Err(PyValueError::new_err(
                "x, y, and dy cannot be empty"
            ));
        }
        
        self.x_values = x;
        self.y_values = y;
        self.dy_values = dy;
        
        // Compute Hermite coefficients
        let (z, coef) = hermite_divided_differences(
            &self.x_values,
            &self.y_values,
            &self.dy_values,
        );
        
        self.z_values = z;
        self.coefficients = coef;
        self.fitted = true;
        
        Ok(())
    }

    /// Get the Hermite polynomial coefficients
    /// 
    /// Returns the divided difference coefficients used in the Newton form
    /// of the Hermite polynomial.
    /// 
    /// Returns
    /// -------
    /// list of float
    ///     Hermite polynomial coefficients in Newton form
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    /// 
    /// Notes
    /// -----
    /// The returned coefficients are for the Newton form with doubled points.
    /// There are 2n coefficients for n data points.
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y, dy) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Evaluate the interpolation at one or more points
    ///
    /// Uses Horner's method for efficient and stable evaluation.
    /// Supports single floats, lists, and NumPy arrays with zero-copy I/O.
    ///
    /// Parameters
    /// ----------
    /// x : float, list of float, or numpy.ndarray
    ///     Point(s) at which to evaluate the interpolation
    ///
    /// Returns
    /// -------
    /// float, list of float, or numpy.ndarray
    ///     Hermite interpolated value(s) at the specified point(s)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If input is neither a float nor a list of floats nor a NumPy array
    ///
    /// Notes
    /// -----
    /// The Hermite polynomial passes through all data points and has the
    /// correct derivative at each point, providing very accurate interpolation
    /// for smooth functions.
    ///
    /// When a NumPy array is passed, evaluation is performed with zero-copy
    /// I/O for maximum performance.
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y, dy) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = hermite_evaluate(&self.z_values, &self.coefficients, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a NumPy array (zero-copy I/O)
        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let x_slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
            {
                let result_slice = unsafe { result_array.as_slice_mut()? };

                // OPTIMIZATION: Cache z and coefficients to avoid repeated lookups
                let z = &self.z_values;
                let coeff = &self.coefficients;
                let n = x_slice.len();
                let mut i = 0;

                // OPTIMIZATION: 2-way unroll evaluation loop for SIMD
                // Process 2 independent points per iteration
                while i + 1 < n {
                    // Evaluate point i
                    result_slice[i] = hermite_evaluate(z, coeff, x_slice[i]);

                    // Evaluate point i+1 (independent - SIMD-friendly)
                    result_slice[i + 1] = hermite_evaluate(z, coeff, x_slice[i + 1]);

                    i += 2;
                }

                // Handle remainder if n is odd
                if i < n {
                    result_slice[i] = hermite_evaluate(z, coeff, x_slice[i]);
                }
            }
            return Ok(result_array.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            // OPTIMIZATION: Cache z and coefficients for better performance
            let z = &self.z_values;
            let coeff = &self.coefficients;
            let n = x_list.len();

            let mut results = Vec::with_capacity(n);

            let mut i = 0;
            // OPTIMIZATION: 2-way unroll evaluation loop
            while i + 1 < n {
                // Evaluate point i
                results.push(hermite_evaluate(z, coeff, x_list[i]));

                // Evaluate point i+1 (independent - SIMD-friendly)
                results.push(hermite_evaluate(z, coeff, x_list[i + 1]));

                i += 2;
            }

            // Handle remainder if n is odd
            if i < n {
                results.push(hermite_evaluate(z, coeff, x_list[i]));
            }

            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, a list of floats, or a NumPy array"
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
                "HermiteInterpolator(fitted with {} points and derivatives)",
                self.x_values.len()
            )
        } else {
            "HermiteInterpolator(not fitted)".to_string()
        }
    }
}