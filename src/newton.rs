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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use rayon::prelude::*;

/// Compute divided differences table
/// 
/// Builds the divided differences table using Newton's divided difference formula.
/// The coefficients are stored in the first column of the table.
/// 
/// # Arguments
/// 
/// * `xs` - Array of x coordinates
/// * `ys` - Array of y coordinates (function values)
/// 
/// # Returns
/// 
/// Vector of divided difference coefficients
fn divided_differences(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut coef = ys.to_vec();
    
    for j in 1..n {
        for i in (j..n).rev() {
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j]);
        }
    }
    coef
}

/// Evaluate Newton polynomial using Horner's method
/// 
/// Efficiently evaluates the Newton polynomial at a point using nested multiplication.
/// 
/// # Arguments
/// 
/// * `xs` - Array of x coordinates
/// * `coef` - Array of divided difference coefficients
/// * `x` - Point at which to evaluate
/// 
/// # Returns
/// 
/// The interpolated value at point x
fn newton_evaluate(xs: &[f64], coef: &[f64], x: f64) -> f64 {
    let n = coef.len();
    let mut result = coef[n - 1];
    
    for i in (0..n - 1).rev() {
        result = result * (x - xs[i]) + coef[i];
    }
    result
}

/// Newton Divided Differences Interpolator
/// 
/// A stateful interpolator that pre-computes divided difference coefficients
/// for efficient repeated evaluation.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored x coordinates of data points
/// * `y_values` - Stored y coordinates of data points
/// * `coefficients` - Pre-computed divided difference coefficients
/// * `fitted` - Whether the interpolator has been fitted
#[pyclass]
pub struct NewtonInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    coefficients: Vec<f64>,
    fitted: bool,
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
            x_values: Vec::new(),
            y_values: Vec::new(),
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points
    /// 
    /// Computes and stores the divided difference coefficients for the Newton
    /// polynomial. This allows for efficient repeated evaluation.
    /// 
    /// Parameters
    /// ----------
    /// x : list of float or numpy.ndarray
    ///     X coordinates of data points
    /// y : list of float or numpy.ndarray
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

        // Check for sorted values and duplicates
        for i in 1..x_vec.len() {
            if x_vec[i] <= x_vec[i - 1] {
                return Err(PyValueError::new_err(
                    "x values must be strictly increasing (sorted and no duplicates)"
                ));
            }
        }

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

        self.x_values = x_vec;
        self.y_values = y_vec;
        
        // Compute Newton coefficients during fitting
        self.coefficients = divided_differences(&self.x_values, &self.y_values);
        
        self.fitted = true;
        Ok(())
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
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Evaluate the interpolation at one or more points
    ///
    /// Uses Horner's method for efficient and numerically stable evaluation.
    ///
    /// Parameters
    /// ----------
    /// x : float, list of float, or numpy.ndarray
    ///     Point(s) at which to evaluate the interpolation
    ///
    /// Returns
    /// -------
    /// float or numpy.ndarray
    ///     Interpolated value(s) at the specified point(s)
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
            let result = newton_evaluate(&self.x_values, &self.coefficients, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Handle numpy array with parallel evaluation
        if let Ok(arr) = x.downcast::<numpy::PyArray1<f64>>() {
            let x_slice = arr.readonly();
            let x_data = x_slice.as_slice()?;

            let results: Vec<f64> = x_data
                .par_iter()
                .map(|&xi| newton_evaluate(&self.x_values, &self.coefficients, xi))
                .collect();

            return Ok(results.to_pyarray(py).into_any().unbind());
        }

        // Handle list of floats with parallel evaluation
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .par_iter()
                .map(|&xi| newton_evaluate(&self.x_values, &self.coefficients, xi))
                .collect();
            return Ok(results.to_pyarray(py).into_any().unbind());
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
                "NewtonInterpolator(fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "NewtonInterpolator(not fitted)".to_string()
        }
    }
}