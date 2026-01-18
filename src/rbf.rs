/// Radial Basis Function (RBF) Interpolation Module
/// 
/// This module implements RBF interpolation, which constructs an interpolant
/// as a weighted sum of radial basis functions centered at data points.
/// 
/// # Mathematical Background
/// 
/// The RBF interpolant has the form:
/// 
/// s(x) = Σᵢ wᵢ φ(||x - xᵢ||, ε)
/// 
/// where:
/// - φ is the radial basis function (kernel)
/// - ||·|| is the Euclidean distance
/// - wᵢ are weights found by solving: Φw = y
/// - ε is the shape parameter
/// 
/// The matrix Φᵢⱼ = φ(||xᵢ - xⱼ||) is symmetric and positive definite
/// for most common kernels.
/// 
/// # Available Kernels
/// 
/// ## 1. Gaussian
/// φ(r) = exp(-ε²r²)
/// - **Properties**: Infinitely smooth (C^∞), compact support in limit
/// - **Shape**: Bell curve, fast decay
/// - **Best for**: Smooth functions, general-purpose
/// - **Epsilon range**: 0.1 - 10
/// 
/// ## 2. Multiquadric
/// φ(r) = √(1 + (εr)²)
/// - **Properties**: Conditionally positive definite
/// - **Shape**: Grows with distance
/// - **Best for**: Scattered data, robust interpolation
/// - **Epsilon range**: 0.5 - 5
/// - **Note**: Most commonly used RBF
/// 
/// ## 3. Inverse Multiquadric
/// φ(r) = 1/√(1 + (εr)²)
/// - **Properties**: Smooth, decreases with distance
/// - **Shape**: Localized, faster decay than Gaussian
/// - **Best for**: Smoother interpolation, less oscillatory
/// - **Epsilon range**: 0.5 - 5
/// 
/// ## 4. Thin Plate Spline
/// φ(r) = r² ln(r)  (r > 0), φ(0) = 0
/// - **Properties**: Minimizes bending energy
/// - **Shape**: Logarithmic growth
/// - **Best for**: Image warping, smooth surfaces
/// - **Epsilon**: Not used (scale-independent)
/// - **Note**: Common in computer graphics
/// 
/// ## 5. Linear
/// φ(r) = r
/// - **Properties**: Simple, continuous
/// - **Shape**: Linear growth
/// - **Best for**: Simple interpolation, debugging
/// - **Epsilon**: Not used
/// 
/// # Epsilon Parameter (Shape Parameter)
/// 
/// Controls the "width" of basis functions:
/// - **Small ε** (< 1): Flat, wide basis functions → smoother, more global
/// - **Large ε** (> 5): Sharp, narrow basis functions → more local, captures detail
/// - **Optimal ε**: Problem-dependent, often ε ≈ 1/(average spacing)
/// 
/// # Characteristics
/// 
/// - **Global method**: All points influence the entire interpolation
/// - **Exact fit**: Passes through all data points
/// - **Meshfree**: No grid required, handles scattered data
/// - **Multidimensional**: Easily extends to 2D, 3D, and higher
/// - **Smooth**: Depends on kernel choice
/// - **Complexity**: O(n³) for weight computation, O(n) per evaluation
/// 
/// # Advantages
/// 
/// - Handles irregularly scattered data naturally
/// - No mesh or grid needed
/// - Works in any dimension
/// - Flexible kernel selection
/// - Theoretically well-understood
/// 
/// # Limitations
/// 
/// - Computational cost for large n (O(n³) solve)
/// - Global method: changing one point affects entire curve
/// - Epsilon selection can be tricky
/// - Can be ill-conditioned for poorly chosen ε
/// 
/// # Use Cases
/// 
/// - Scattered data interpolation
/// - Multidimensional problems (2D, 3D surfaces)
/// - Irregular point distributions
/// - Computer graphics and image processing
/// - Geospatial data (elevation maps)
/// - Meshfree numerical methods
/// - Machine learning (RBF networks)
/// 
/// # Kernel Selection Guide
/// 
/// | Data Characteristics | Recommended Kernel |
/// |---------------------|-------------------|
/// | Smooth, general | Gaussian |
/// | Scattered, robust | Multiquadric |
/// | Very smooth | Inverse Multiquadric |
/// | Image/surface | Thin Plate Spline |
/// | Simple/debug | Linear |
/// 
/// # Examples
/// 
/// ```python
/// from interlib import RBFInterpolator
/// 
/// # Gaussian kernel (most common)
/// rbf = RBFInterpolator(kernel="gaussian", epsilon=1.0)
/// x = [0.0, 1.0, 2.0, 3.0]
/// y = [0.0, 1.0, 0.5, 2.0]
/// rbf.fit(x, y)
/// result = rbf(1.5)
/// 
/// # Thin plate spline (for smooth surfaces)
/// rbf_tps = RBFInterpolator(kernel="thin_plate_spline", epsilon=1.0)
/// rbf_tps.fit(x, y)
/// 
/// # Get weights
/// weights = rbf.get_weights()
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use rayon::prelude::*;

/// RBF kernel types with their evaluation functions
#[derive(Clone, Copy, Debug)]
pub enum RBFKernel {
    Gaussian,
    Multiquadric,
    InverseMultiquadric,
    ThinPlateSpline,
    Linear,
}

impl RBFKernel {
    /// Evaluate the RBF kernel at distance r with shape parameter epsilon
    /// 
    /// # Arguments
    /// 
    /// * `r` - Distance from center
    /// * `epsilon` - Shape parameter (not used for TPS and Linear)
    /// 
    /// # Returns
    /// 
    /// Kernel value φ(r, ε)
    fn evaluate(&self, r: f64, epsilon: f64) -> f64 {
        match self {
            RBFKernel::Gaussian => (-epsilon * epsilon * r * r).exp(),
            RBFKernel::Multiquadric => (1.0 + (epsilon * r).powi(2)).sqrt(),
            RBFKernel::InverseMultiquadric => 1.0 / (1.0 + (epsilon * r).powi(2)).sqrt(),
            RBFKernel::ThinPlateSpline => {
                if r == 0.0 {
                    0.0
                } else {
                    r * r * (r.ln())
                }
            }
            RBFKernel::Linear => r,
        }
    }
}

/// Solve linear system using Gaussian elimination with partial pivoting
/// 
/// Solves the system Φw = y for weights w.
/// 
/// # Arguments
/// 
/// * `a` - Coefficient matrix (modified in-place)
/// * `b` - Right-hand side (modified in-place)
/// 
/// # Returns
/// 
/// Solution vector or error message if singular
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    
    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();
        
        for i in k + 1..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_idx = i;
            }
        }
        
        // Swap rows if needed
        if max_idx != k {
            a.swap(k, max_idx);
            b.swap(k, max_idx);
        }
        
        // Check for singular matrix
        if a[k][k].abs() < 1e-12 {
            return Err("Matrix is singular or nearly singular".to_string());
        }
        
        // Eliminate column
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    
    Ok(x)
}

/// Compute RBF interpolation weights
/// 
/// Solves Φw = y where Φᵢⱼ = φ(||xᵢ - xⱼ||).
/// 
/// # Arguments
/// 
/// * `x_values` - Data point locations
/// * `y_values` - Function values at data points
/// * `kernel` - RBF kernel type
/// * `epsilon` - Shape parameter
/// 
/// # Returns
/// 
/// Weight vector w or error message
fn compute_rbf_weights(
    x_values: &[f64],
    y_values: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> Result<Vec<f64>, String> {
    let n = x_values.len();
    
    // Build the RBF matrix Φ
    let mut matrix = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            let r = (x_values[i] - x_values[j]).abs();
            matrix[i][j] = kernel.evaluate(r, epsilon);
        }
    }
    
    solve_linear_system(matrix, y_values.to_vec())
}

/// Evaluate RBF interpolation at a point
/// 
/// Computes s(x) = Σᵢ wᵢ φ(||x - xᵢ||).
/// 
/// # Arguments
/// 
/// * `x_values` - Data point locations
/// * `weights` - Pre-computed weights
/// * `kernel` - RBF kernel type
/// * `epsilon` - Shape parameter
/// * `x` - Evaluation point
/// 
/// # Returns
/// 
/// Interpolated value at x
fn rbf_evaluate(
    x_values: &[f64],
    weights: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
    x: f64,
) -> f64 {
    let mut result = 0.0;
    
    for i in 0..x_values.len() {
        let r = (x - x_values[i]).abs();
        result += weights[i] * kernel.evaluate(r, epsilon);
    }
    
    result
}

/// Radial Basis Function Interpolator
/// 
/// Global interpolator using weighted radial basis functions.
/// 
/// # Attributes
/// 
/// * `x_values` - Stored data point locations
/// * `y_values` - Stored function values
/// * `weights` - Computed interpolation weights
/// * `kernel` - Selected RBF kernel
/// * `epsilon` - Shape parameter
/// * `fitted` - Whether weights have been computed
#[pyclass]
pub struct RBFInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    weights: Vec<f64>,
    kernel: RBFKernel,
    epsilon: f64,
    fitted: bool,
}

#[pymethods]
impl RBFInterpolator {
    /// Create a new RBF interpolator
    /// 
    /// Parameters
    /// ----------
    /// kernel : str, default="gaussian"
    ///     RBF kernel type. Options:
    ///     - "gaussian": Smooth, general purpose
    ///     - "multiquadric": Robust, most common
    ///     - "inverse_multiquadric": Very smooth
    ///     - "thin_plate_spline": Minimal bending
    ///     - "linear": Simple, for debugging
    /// epsilon : float, default=1.0
    ///     Shape parameter (not used for thin_plate_spline, linear)
    ///     - Smaller values → flatter, smoother
    ///     - Larger values → sharper, more local
    ///     - Typical range: 0.1 to 10
    /// 
    /// Returns
    /// -------
    /// RBFInterpolator
    ///     A new, unfitted interpolator instance
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If kernel name is invalid or epsilon is non-positive
    #[new]
    #[pyo3(signature = (kernel="gaussian", epsilon=1.0))]
    pub fn new(kernel: &str, epsilon: f64) -> PyResult<Self> {
        let kernel_type = match kernel.to_lowercase().as_str() {
            "gaussian" => RBFKernel::Gaussian,
            "multiquadric" => RBFKernel::Multiquadric,
            "inverse_multiquadric" | "inverse multiquadric" => RBFKernel::InverseMultiquadric,
            "thin_plate_spline" | "thin plate spline" => RBFKernel::ThinPlateSpline,
            "linear" => RBFKernel::Linear,
            _ => return Err(PyValueError::new_err(
                format!("Unknown kernel type: '{}'. Available: 'gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline', 'linear'", kernel)
            )),
        };
        
        if epsilon <= 0.0 {
            return Err(PyValueError::new_err("epsilon must be positive"));
        }
        
        Ok(RBFInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            weights: Vec::new(),
            kernel: kernel_type,
            epsilon,
            fitted: false,
        })
    }

    /// Fit the RBF interpolator
    ///
    /// Computes interpolation weights by solving the RBF system Φw = y.
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
    ///     If x and y have different lengths or are empty
    ///     If weight computation fails (singular matrix)
    ///     If x values are not sorted or contain duplicates
    ///
    /// Notes
    /// -----
    /// Complexity is O(n³) due to solving the linear system.
    /// For large datasets (n > 1000), consider other methods.
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

        // Check for sorted values and duplicates
        for i in 1..x_vec.len() {
            if x_vec[i] <= x_vec[i - 1] {
                return Err(PyValueError::new_err(
                    "x values must be strictly increasing (sorted and no duplicates)"
                ));
            }
        }

        self.x_values = x_vec;
        self.y_values = y_vec;

        // Compute RBF weights
        match compute_rbf_weights(&self.x_values, &self.y_values, self.kernel, self.epsilon) {
            Ok(weights) => {
                self.weights = weights;
                self.fitted = true;
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!("Failed to compute weights: {}", e))),
        }
    }

    /// Get the RBF interpolation weights
    /// 
    /// Returns
    /// -------
    /// list of float
    ///     Weight vector [w₀, w₁, ..., wₙ₋₁]
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted
    pub fn get_weights(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.weights.clone())
    }

    /// Evaluate the interpolation at one or more points
    ///
    /// Parameters
    /// ----------
    /// x : float, numpy.ndarray, or list of float
    ///     Point(s) at which to evaluate
    ///
    /// Returns
    /// -------
    /// float or numpy.ndarray
    ///     Interpolated value(s)
    ///     Returns numpy array if input is numpy array or list
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If not fitted or invalid input type
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // NumPy array with parallel evaluation
        if let Ok(arr) = x.downcast::<numpy::PyArray1<f64>>() {
            let x_slice = arr.readonly();
            let x_data = x_slice.as_slice()?;

            let results: Vec<f64> = x_data
                .par_iter()
                .map(|&xi| rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, xi))
                .collect();

            return Ok(results.to_pyarray(py).into_any().unbind());
        }

        // List of floats with parallel evaluation
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .par_iter()
                .map(|&xi| rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, xi))
                .collect();
            return Ok(results.to_pyarray(py).into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float, numpy array, or a list of floats"
        ))
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        let kernel_name = match self.kernel {
            RBFKernel::Gaussian => "gaussian",
            RBFKernel::Multiquadric => "multiquadric",
            RBFKernel::InverseMultiquadric => "inverse_multiquadric",
            RBFKernel::ThinPlateSpline => "thin_plate_spline",
            RBFKernel::Linear => "linear",
        };
        
        if self.fitted {
            format!(
                "RBFInterpolator(kernel='{}', epsilon={:.2}, fitted with {} points)",
                kernel_name,
                self.epsilon,
                self.x_values.len()
            )
        } else {
            format!(
                "RBFInterpolator(kernel='{}', epsilon={:.2}, not fitted)",
                kernel_name,
                self.epsilon
            )
        }
    }
}