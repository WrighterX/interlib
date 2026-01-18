/// Lagrange Interpolation Module
/// 
/// This module implements Lagrange polynomial interpolation, which constructs
/// a polynomial that passes exactly through all given data points.
/// 
/// # Mathematical Background
/// 
/// Given n points (x₀, y₀), (x₁, y₁), ..., (xₙ₋₁, yₙ₋₁), the Lagrange polynomial is:
/// 
/// P(x) = Σ yᵢ · Lᵢ(x)
/// 
/// where Lᵢ(x) is the i-th Lagrange basis polynomial:
/// 
/// Lᵢ(x) = Π(j≠i) (x - xⱼ) / (xᵢ - xⱼ)
/// 
/// # Characteristics
/// 
/// - **Exact fit**: Passes through all data points exactly
/// - **Global method**: All points influence the entire curve
/// - **Degree**: Polynomial degree is n-1 for n points
/// - **Oscillations**: Can exhibit Runge's phenomenon with many uniformly-spaced points
/// - **Complexity**: O(n²) evaluation time
/// 
/// # Use Cases
/// 
/// - Small number of data points (< 10-15)
/// - Exact interpolation required
/// - Mathematical function approximation
/// - Not recommended for large datasets or uniformly-spaced points
/// 
/// # Examples
/// 
/// ```python
/// from interlib import LagrangeInterpolator
/// 
/// # Create interpolator
/// interp = LagrangeInterpolator()
/// 
/// # Fit with data points
/// x = [0.0, 1.0, 2.0, 3.0]
/// y = [0.0, 1.0, 4.0, 9.0]
/// interp.fit(x, y)
/// 
/// # Evaluate at new points
/// result = interp(1.5)  # Single point
/// results = interp([1.5, 2.5])  # Multiple points
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use rayon::prelude::*;

/// Core Lagrange interpolation function
/// 
/// Evaluates the Lagrange polynomial at a single point x using the basis polynomial approach.
/// 
/// # Arguments
/// 
/// * `x_values` - Array of x coordinates of data points
/// * `y_values` - Array of y coordinates of data points
/// * `x` - Point at which to evaluate the polynomial
/// 
/// # Returns
/// 
/// The interpolated value at point x
fn lagrange_interpolation(x_values: &[f64], y_values: &[f64], x: f64) -> f64 {
    let n = x_values.len();
    let mut result = 0.0;
    
    for i in 0..n {
        let mut term = y_values[i];
        for j in 0..n {
            if i != j {
                term *= (x - x_values[j]) / (x_values[i] - x_values[j]);
            }
        }
        result += term;
    }
    
    result
}

/// Lagrange Polynomial Interpolator
/// 
/// A stateful interpolator that fits a Lagrange polynomial through data points
/// and allows evaluation at arbitrary points.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of fitted data points
/// * `y_values` - Stored y coordinates of fitted data points
/// * `fitted` - Whether the interpolator has been fitted with data
#[pyclass]
pub struct LagrangeInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}
#[pymethods]
impl LagrangeInterpolator {
    /// Create a new Lagrange interpolator
    /// 
    /// Returns
    /// -------
    /// LagrangeInterpolator
    ///     A new, unfitted interpolator instance
    #[new]
    pub fn new() -> Self {
        LagrangeInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points
    ///
    /// Stores the data points for later evaluation. The Lagrange polynomial
    /// is constructed implicitly and evaluated on-demand.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray or list of float
    ///     X coordinates of data points
    /// y : numpy.ndarray or list of float
    ///     Y coordinates of data points
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If x and y have different lengths or if either is empty
    ///     If x or y contain duplicate or unsorted values
    ///
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    pub fn fit(&mut self, x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<()> {
        // Try to extract x as numpy array first (zero-copy read), then as Vec
        let x_vec: Vec<f64> = if let Ok(arr) = x.downcast::<numpy::PyArray1<f64>>() {
            arr.readonly().as_slice()?.to_vec()
        } else if let Ok(vec) = x.extract::<Vec<f64>>() {
            vec
        } else {
            return Err(PyValueError::new_err(
                "x must be a numpy array or list of floats"
            ));
        };

        // Try to extract y as numpy array first (zero-copy read), then as Vec
        let y_vec: Vec<f64> = if let Ok(arr) = y.downcast::<numpy::PyArray1<f64>>() {
            arr.readonly().as_slice()?.to_vec()
        } else if let Ok(vec) = y.extract::<Vec<f64>>() {
            vec
        } else {
            return Err(PyValueError::new_err(
                "y must be a numpy array or list of floats"
            ));
        };

        // Validation
        if x_vec.len() != y_vec.len() {
            return Err(PyValueError::new_err(
                "x and y must have the same length"
            ));
        }
        if x_vec.is_empty() {
            return Err(PyValueError::new_err(
                "x and y cannot be empty"
            ));
        }

        // Check for sorted values
        for i in 1..x_vec.len() {
            if x_vec[i] <= x_vec[i - 1] {
                return Err(PyValueError::new_err(
                    "x values must be strictly increasing (sorted and no duplicates)"
                ));
            }
        }

        self.x_values = x_vec;
        self.y_values = y_vec;
        self.fitted = true;
        Ok(())
    }

    /// Evaluate the interpolation at one or more points
    ///
    /// Parameters
    /// ----------
    /// x : float, numpy.ndarray, or list of float
    ///     Point(s) at which to evaluate the interpolation
    ///
    /// Returns
    /// -------
    /// float or numpy.ndarray
    ///     Interpolated value(s) at the specified point(s)
    ///     Returns numpy array if input is numpy array or list
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If input is neither a float, numpy array, nor a list of floats
    ///
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp(1.5)  # Single point
    /// 2.25
    /// >>> interp([0.5, 1.5])  # Multiple points
    /// array([0.25, 2.25])
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = lagrange_interpolation(&self.x_values, &self.y_values, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as numpy array (zero-copy read)
        if let Ok(arr) = x.downcast::<numpy::PyArray1<f64>>() {
            let x_slice = arr.readonly();
            let x_data = x_slice.as_slice()?;

            // Use parallel evaluation for arrays with Rayon
            let results: Vec<f64> = x_data
                .par_iter()
                .map(|&xi| lagrange_interpolation(&self.x_values, &self.y_values, xi))
                .collect();

            return Ok(results.to_pyarray(py).into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            // Use parallel evaluation for lists with Rayon
            let results: Vec<f64> = x_list
                .par_iter()
                .map(|&xi| lagrange_interpolation(&self.x_values, &self.y_values, xi))
                .collect();
            return Ok(results.to_pyarray(py).into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, numpy array, or a list of floats"
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
                "LagrangeInterpolator(fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "LagrangeInterpolator(not fitted)".to_string()
        }
    }
}