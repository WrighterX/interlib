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
use crate::linear_core::LinearCore;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    core: LinearCore,
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
            core: LinearCore::new(),
        }
    }

    /// Fit the interpolator with data points
    ///
    /// Stores the data points for later evaluation. If x values are not sorted,
    /// they will be automatically sorted along with the corresponding y values.
    /// No pre-computation is needed for linear interpolation.
    ///
    /// Parameters
    /// ----------
    /// x : list of float
    ///     X coordinates of data points (will be sorted if necessary)
    /// y : list of float
    ///     Y coordinates of data points
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If x and y have different lengths
    ///     If x or y is empty
    ///
    /// Notes
    /// -----
    /// If x values are not in strictly increasing order, both x and y arrays
    /// will be automatically sorted by x values. This means the input arrays
    /// may be reordered, but the final interpolation remains mathematically correct.
    ///
    /// Examples
    /// --------
    /// >>> interp = LinearInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.core.fit(x, y).map_err(PyValueError::new_err)
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
        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = self
                .core
                .evaluate_single(single_x)
                .map_err(PyValueError::new_err)?;
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a NumPy array (zero-copy, most efficient)
        if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
            let x_slice = arr.as_slice()?;
            let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
            {
                let result_slice = unsafe { result_array.as_slice_mut()? };
                self.core
                    .fill_many(x_slice, result_slice)
                    .map_err(PyValueError::new_err)?;
            }
            return Ok(result_array.into_any().unbind());
        }

        // Try to extract as a list of floats (with 2-way unrolling)
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results = self
                .core
                .evaluate_many(&x_list)
                .map_err(PyValueError::new_err)?;
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, list of floats, or NumPy array",
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
        self.core.update_y(y).map_err(PyValueError::new_err)
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
        self.core
            .add_point(x_new, y_new)
            .map_err(PyValueError::new_err)
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
