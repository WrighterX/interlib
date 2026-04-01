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
use crate::quadratic_core::QuadraticCore;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};

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
    core: QuadraticCore,
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
            core: QuadraticCore::new(),
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
        self.core.fit(x, y).map_err(PyValueError::new_err)
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
        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = self.core.evaluate_single(single_x).map_err(PyValueError::new_err)?;
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a NumPy array (zero-copy, most efficient)
        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let x_slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
            {
                let result_slice = unsafe { result_array.as_slice_mut()? };
                self.core.fill_many(x_slice, result_slice).map_err(PyValueError::new_err)?;
            }
            return Ok(result_array.into_any().unbind());
        }

        // Try to extract as a list of floats (with 2-way unrolling)
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results = self.core.evaluate_many(&x_list).map_err(PyValueError::new_err)?;
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
        self.core.repr()
    }
}
