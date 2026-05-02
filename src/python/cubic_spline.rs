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
/// # Advantages over Linear
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
use crate::core::cubic_spline_core::CubicSplineCore;
use crate::python::pywrap_macros::py_call_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Natural Cubic Spline Interpolator
///
/// A stateful interpolator that pre-computes cubic spline segments
/// for smooth C² continuous interpolation.
///
/// # Attributes
///
/// * `x_values` - Stored x coordinates of data points
/// * `segments` - Pre-computed cubic polynomial segments
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct CubicSplineInterpolator {
    core: CubicSplineCore,
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
            core: CubicSplineCore::new(),
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
        self.core.fit(x, y).map_err(PyValueError::new_err)
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
        self.core.num_segments().map_err(PyValueError::new_err)
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
        py_call_impl(&self.core, py, x)
    }

    /// String representation of the interpolator
    ///
    /// Returns
    /// -------
    /// str
    ///     Description of the interpolator state including number of segments
    pub fn __repr__(&self) -> String {
        self.core.repr()
    }
}
