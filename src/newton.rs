/// Newton Divided Differences Interpolation Module
///
/// This module implements Newton's divided differences interpolation method,
/// which is mathematically equivalent to Lagrange interpolation but offers
/// more efficient computation and easier incremental construction.
///
/// # Mathematical Background
///
/// The Newton interpolation polynomial is:
///
/// P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
///
/// where f[x₀,x₁,...,xₖ] are divided differences:
///
/// f[xᵢ,xᵢ₊₁,...,xⱼ] = (f[xᵢ₊₁,...,xⱼ] - f[xᵢ,...,xⱼ₋₁]) / (xⱼ - xᵢ)
///
/// # Characteristics
///
/// - **Exact fit**: Passes through all data points (same as Lagrange)
/// - **Efficient**: Better computational efficiency than Lagrange
/// - **Incremental**: Easy to add new points without recomputation
/// - **Degree**: Polynomial degree is n-1 for n points
/// - **Complexity**: O(n²) for coefficient computation, O(n) for evaluation
///
/// # Advantages over Lagrange
///
/// - Faster evaluation using Horner's method
/// - Coefficients computed once during fitting
/// - Easier to implement adaptive interpolation
/// - Better numerical stability
///
/// # Use Cases
///
/// - Same as Lagrange but with better performance
/// - Preferred for repeated evaluations
/// - Adaptive interpolation schemes
/// - Building interpolation tables
///
/// # Examples
///
/// ```python
/// from interlib import NewtonInterpolator
///
/// # Create interpolator
/// interp = NewtonInterpolator()
///
/// # Fit with data points
/// x = [0.0, 1.0, 2.0, 3.0]
/// y = [0.0, 1.0, 4.0, 9.0]
/// interp.fit(x, y)
///
/// # Get the divided difference coefficients
/// coeffs = interp.get_coefficients()
///
/// # Evaluate at new points
/// result = interp(1.5)
/// ```
use crate::newton_core::NewtonCore;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Newton Divided Differences Interpolator
///
/// A stateful interpolator that pre-computes divided difference coefficients
/// for efficient repeated evaluation.
///
/// # Attributes
///
/// * `x_values` - Stored x coordinates of data points
/// * `coefficients` - Pre-computed divided difference coefficients
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct NewtonInterpolator {
    core: NewtonCore,
}

#[pymethods]
impl NewtonInterpolator {
    /// Create a new Newton interpolator
    ///
    /// Returns
    /// -------
    /// NewtonInterpolator
    ///     A new, unfitted interpolator instance
    #[new]
    pub fn new() -> Self {
        NewtonInterpolator {
            core: NewtonCore::new(),
        }
    }

    /// Fit the interpolator with data points
    ///
    /// Computes and stores the divided difference coefficients for the Newton
    /// polynomial. This allows for efficient repeated evaluation.
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
    ///     If x and y have different lengths or if either is empty
    ///
    /// Notes
    /// -----
    /// The divided differences are computed once during fitting, making
    /// subsequent evaluations more efficient than Lagrange interpolation.
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        self.core.fit(x, y).map_err(PyValueError::new_err)
    }

    /// Get the Newton polynomial divided difference coefficients
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Divided difference coefficients [f[x₀], f[x₀,x₁], f[x₀,x₁,x₂], ...]
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///
    /// Notes
    /// -----
    /// The i-th coefficient multiplies the term (x-x₀)(x-x₁)...(x-xᵢ₋₁)
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        self.core.get_coefficients().map_err(PyValueError::new_err)
    }

    /// Evaluate the interpolation at one or more points
    ///
    /// Uses Horner's method for efficient and numerically stable evaluation.
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
