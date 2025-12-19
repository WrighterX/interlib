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
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
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
    ///     Interpolated value(s) at the specified point(s)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    ///     If input is neither a float nor a list of floats
    /// 
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp(1.5)  # Single point
    /// 2.25
    /// >>> interp([0.5, 1.5])  # Multiple points
    /// [0.25, 2.25]
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

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| lagrange_interpolation(&self.x_values, &self.y_values, xi))
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
                "LagrangeInterpolator(fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "LagrangeInterpolator(not fitted)".to_string()
        }
    }
}