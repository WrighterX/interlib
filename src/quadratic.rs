// TODO: Replace cramer with gauss?

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
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};

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
#[inline]
fn solve_quadratic_coefficients(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64) {
    let x0_sq = x0 * x0;
    let x1_sq = x1 * x1;
    let x2_sq = x2 * x2;

    // Main Determinant (Vandermonde-like matrix)
    let det = 1.0 * (x1 * x2_sq - x2 * x1_sq)
            - x0 * (1.0 * x2_sq - 1.0 * x1_sq)
            + x0_sq * (1.0 * x2 - 1.0 * x1);

    if det.abs() < 1e-12 {
        // Fallback to linear if points are nearly collinear
        let slope = (y1 - y0) / (x1 - x0);
        let intercept = y0 - slope * x0;
        return (intercept, slope, 0.0);
    }

    // Using a cleaner expansion pattern for clarity and correctness:
    let da = y0 * (x1 * x2_sq - x2 * x1_sq) - y1 * (x0 * x2_sq - x2 * x0_sq) + y2 * (x0 * x1_sq - x1 * x0_sq);
    let db = 1.0 * (y1 * x2_sq - y2 * x1_sq) - 1.0 * (y0 * x2_sq - y2 * x0_sq) + 1.0 * (y0 * x1_sq - y1 * x0_sq);

    // CORRECTED det_c: replace 3rd column [x0^2, x1^2, x2^2] with [y0, y1, y2]
    let dc = 1.0 * (x1 * y2 - x2 * y1)
           - x0 * (1.0 * y2 - 1.0 * y1)
           + y0 * (1.0 * x2 - 1.0 * x1);

    (da / det, db / det, dc / det)
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
#[inline]
fn eval_quadratic(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a + b * x + c * x * x
}

/// Piecewise Quadratic Interpolator
///
/// A stateful interpolator that performs piecewise quadratic interpolation
/// using overlapping triplets of data points.
///
/// Coefficients are pre-computed at fit time and stored in a flat Vec<f64>
/// for cache-friendly access during evaluation.
///
/// # Attributes
///
/// * `x_values` - Stored x coordinates of data points
/// * `coefficients` - Pre-computed (a, b, c) per segment, flat layout [a0,b0,c0, a1,b1,c1, ...]
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct QuadraticInterpolator {
    x_values: Vec<f64>,
    coefficients: Vec<f64>,  // flat: [a0, b0, c0, a1, b1, c1, ...]
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
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points
    ///
    /// Pre-computes quadratic coefficients for all segments at fit time,
    /// enabling O(log n) evaluation via binary search + direct lookup.
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

        let n = x.len();
        let n_segments = n - 2;  // overlapping triplets: [0,1,2], [1,2,3], ..., [n-3,n-2,n-1]

        // Pre-compute all (a, b, c) coefficients — flat layout for cache locality
        let mut coefficients = Vec::with_capacity(n_segments * 3);
        for i in 0..n_segments {
            let (a, b, c) = solve_quadratic_coefficients(
                x[i],     y[i],
                x[i + 1], y[i + 1],
                x[i + 2], y[i + 2],
            );
            coefficients.push(a);
            coefficients.push(b);
            coefficients.push(c);
        }

        self.x_values = x;
        self.coefficients = coefficients;
        self.fitted = true;
        Ok(())
    }

    /// Evaluate the interpolation at one or more points
    ///
    /// Uses binary search to find the interval in O(log n), then looks up
    /// pre-computed coefficients for O(1) evaluation.
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
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = self.eval_single(single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a NumPy array (zero-copy, most efficient)
        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let x_slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
            let result_slice = unsafe { result_array.as_slice_mut()? };
            let n = x_slice.len();
            let mut i = 0;

            // 2-way loop unrolling for batch evaluation
            while i + 1 < n {
                result_slice[i]     = self.eval_single(x_slice[i]);
                result_slice[i + 1] = self.eval_single(x_slice[i + 1]);
                i += 2;
            }

            if i < n {
                result_slice[i] = self.eval_single(x_slice[i]);
            }

            return Ok(result_array.into_any().unbind());
        }

        // Try to extract as a list of floats (with 2-way unrolling)
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let n = x_list.len();
            let mut results = Vec::with_capacity(n);
            let mut i = 0;

            while i + 1 < n {
                results.push(self.eval_single(x_list[i]));
                results.push(self.eval_single(x_list[i + 1]));
                i += 2;
            }

            if i < n {
                results.push(self.eval_single(x_list[i]));
            }

            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, list of floats, or NumPy array"
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

impl QuadraticInterpolator {
    /// Evaluate at a single point using pre-computed coefficients and binary search.
    #[inline]
    fn eval_single(&self, x: f64) -> f64 {
        let x_values = &self.x_values;
        let n = x_values.len();
        let n_segments = n - 2;

        // Binary search: find first index where x_values[i] > x
        let pos = x_values.partition_point(|&xi| xi <= x);

        // Map position to segment index, clamped to valid range [0, n_segments - 1]
        // pos=0 → before first point → use segment 0
        // pos=n → after last point  → use segment n_segments-1
        // pos=i (1..n-1) → in interval [i-1, i] → use segment clamped(i-2, 0, n_segments-1)
        let seg_idx = pos.saturating_sub(2).min(n_segments - 1);

        let base = seg_idx * 3;
        let a = self.coefficients[base];
        let b = self.coefficients[base + 1];
        let c = self.coefficients[base + 2];

        eval_quadratic(a, b, c, x)
    }
}
