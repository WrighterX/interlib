/// Chebyshev Polynomial Interpolation Module
/// 
/// This module implements Chebyshev interpolation using optimally-placed nodes
/// to achieve spectral accuracy and avoid Runge's phenomenon.
/// 
/// # Mathematical Background
/// 
/// ## Chebyshev Nodes (Type 1)
/// 
/// On the standard interval [-1, 1]:
/// xₖ = cos((2k + 1)π / 2n)  for k = 0, 1, ..., n-1
/// 
/// These are the roots of the n-th Chebyshev polynomial Tₙ(x).
/// 
/// ## Chebyshev Polynomials
/// 
/// Defined by the recurrence relation:
/// - T₀(x) = 1
/// - T₁(x) = x
/// - Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
/// 
/// Or directly: Tₙ(x) = cos(n·arccos(x)) for x ∈ [-1, 1]
/// 
/// ## Interpolant Form
/// 
/// P(x) = Σₖ cₖ Tₖ(x)
/// 
/// where coefficients cₖ are computed using discrete cosine transform (DCT).
/// 
/// # Why Chebyshev Nodes?
/// 
/// ## Problem with Uniform Points
/// 
/// High-degree polynomial interpolation at uniformly spaced points leads to:
/// - Large oscillations near boundaries (Runge's phenomenon)
/// - Exponentially growing errors
/// - Unstable behavior
/// 
/// ## Chebyshev Solution
/// 
/// - Nodes cluster near boundaries (where errors typically occur)
/// - Minimizes maximum interpolation error (minimax property)
/// - Achieves near-optimal polynomial approximation
/// - Stable for high polynomial degrees
/// 
/// # Characteristics
/// 
/// - **Optimal nodes**: Minimizes interpolation error
/// - **No Runge phenomenon**: Stable even for high degrees
/// - **Spectral accuracy**: Exponential convergence for smooth functions
/// - **Fixed interval**: Works on transformed interval [a, b]
/// - **Fast evaluation**: Clenshaw algorithm is O(n)
/// - **Complexity**: O(n log n) with FFT, O(n²) without
/// 
/// # Error Behavior
/// 
/// For smooth functions:
/// - Error decreases exponentially with n
/// - Much faster convergence than uniform points
/// - Error bounded by Lebesgue constant O(log n)
/// 
/// For non-smooth functions:
/// - Convergence slows at discontinuities
/// - Still better than uniform points
/// 
/// # Evaluation Methods
/// 
/// ## Clenshaw Algorithm (Default)
/// - Numerically stable recursive evaluation
/// - O(n) complexity
/// - Uses backward recursion
/// - Recommended for most applications
/// 
/// ## Direct Polynomial Evaluation
/// - Evaluates each Chebyshev polynomial explicitly
/// - O(n²) complexity
/// - Useful for understanding/debugging
/// - Can be less stable numerically
/// 
/// # Advantages
/// 
/// - Solves Runge's phenomenon
/// - Near-optimal approximation
/// - Well-understood theory
/// - Stable numerically
/// - Efficient evaluation
/// - Scales to high degrees
/// 
/// # Limitations
/// 
/// - Fixed interval [a, b] required
/// - Must evaluate at specific nodes
/// - Not adaptive to local features
/// - Requires smooth function for best results
/// 
/// # Use Cases
/// 
/// - High-accuracy function approximation
/// - Spectral methods in numerical analysis
/// - Function libraries and lookup tables
/// - When Runge's phenomenon is a concern
/// - Smooth functions on bounded intervals
/// - Polynomial chaos expansions
/// - Approximation theory research
/// 
/// # When NOT to Use
/// 
/// - Scattered/irregular data points
/// - Discontinuous functions
/// - Adaptive mesh needed
/// - Unknown interval boundaries
/// - Non-smooth data
/// 
/// # Comparison with Other Methods
/// 
/// | Property | Chebyshev | Lagrange (uniform) | Cubic Spline |
/// |----------|-----------|-------------------|--------------|
/// | Runge's phenomenon | No | Yes | No |
/// | Max error growth | O(log n) | Exponential | O(h⁴) |
/// | Convergence rate | Exponential | Algebraic | O(h⁴) |
/// | Node placement | Fixed (optimal) | Any | Any |
/// | Smoothness | C^∞ | C^∞ | C² |
/// 
/// # Examples
/// 
/// ```python
/// from interlib import ChebyshevInterpolator
/// import math
/// 
/// # Create interpolator for [0, 2π] with 15 nodes
/// cheb = ChebyshevInterpolator(n_points=15, x_min=0.0, x_max=2*math.pi)
/// 
/// # Get Chebyshev nodes
/// nodes = cheb.get_nodes()
/// 
/// # Evaluate function at nodes
/// y_values = [math.sin(x) for x in nodes]
/// 
/// # Fit
/// cheb.fit(y_values)
/// 
/// # Or use convenience method with function
/// cheb.fit_function(math.sin)
/// 
/// # Evaluate
/// result = cheb(1.5)  # Very accurate for smooth functions
/// 
/// # Get coefficients
/// coeffs = cheb.get_coefficients()
/// 
/// # Switch evaluation method
/// cheb.set_method(use_clenshaw=False)  # Use direct evaluation
/// ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::f64::consts::PI;
use pyo3_stub_gen::{derive::gen_stub_pymethods, derive::gen_stub_pyclass, define_stub_info_gatherer};

/// Generate Chebyshev nodes of the first kind on interval [a, b]
/// 
/// Computes xₖ = cos((2k+1)π/2n) on [-1, 1] and transforms to [a, b].
/// 
/// # Arguments
/// 
/// * `n` - Number of nodes
/// * `a` - Left endpoint
/// * `b` - Right endpoint
/// 
/// # Returns
/// 
/// Vector of Chebyshev nodes in [a, b]
fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    let mut nodes = Vec::with_capacity(n);
    
    for k in 0..n {
        // Chebyshev nodes on [-1, 1]
        let x = ((2 * k + 1) as f64 * PI / (2 * n) as f64).cos();
        
        // Transform to [a, b]
        let x_transformed = 0.5 * (b - a) * x + 0.5 * (b + a);
        nodes.push(x_transformed);
    }
    
    nodes
}

/// Transform x from [a, b] to [-1, 1]
/// 
/// Linear transformation for evaluating Chebyshev polynomials.
fn transform_to_standard(x: f64, a: f64, b: f64) -> f64 {
    2.0 * (x - a) / (b - a) - 1.0
}

/// Compute Chebyshev polynomial Tₙ(x) using recurrence relation
/// 
/// Uses the three-term recurrence:
/// T₀(x) = 1, T₁(x) = x, Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
/// 
/// # Arguments
/// 
/// * `n` - Polynomial degree
/// * `x` - Evaluation point in [-1, 1]
/// 
/// # Returns
/// 
/// Value of Tₙ(x)
fn chebyshev_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    
    let mut t_prev2 = 1.0; // T₀
    let mut t_prev1 = x;   // T₁
    let mut t_n = 0.0;
    
    for _ in 2..=n {
        t_n = 2.0 * x * t_prev1 - t_prev2;
        t_prev2 = t_prev1;
        t_prev1 = t_n;
    }
    
    t_n
}

/// Compute Chebyshev coefficients using DCT-like approach
/// 
/// Computes cₖ = (2/n) Σⱼ yⱼ cos(kθⱼ) where θⱼ = (2j+1)π/2n
/// (with special case c₀ = (1/n) Σⱼ yⱼ)
/// 
/// # Arguments
/// 
/// * `y_values` - Function values at Chebyshev nodes
/// 
/// # Returns
/// 
/// Chebyshev polynomial coefficients
fn compute_chebyshev_coefficients(y_values: &[f64]) -> Vec<f64> {
    let n = y_values.len();
    let mut coefficients = vec![0.0; n];
    
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let theta = ((2 * j + 1) as f64 * k as f64 * PI) / (2 * n) as f64;
            sum += y_values[j] * theta.cos();
        }
        
        if k == 0 {
            coefficients[k] = sum / n as f64;
        } else {
            coefficients[k] = 2.0 * sum / n as f64;
        }
    }
    
    coefficients
}

/// Evaluate Chebyshev series using Clenshaw algorithm
/// 
/// Numerically stable evaluation using backward recursion.
/// Computes P(x) = Σₖ cₖTₖ(x) efficiently.
/// 
/// # Arguments
/// 
/// * `coefficients` - Chebyshev coefficients
/// * `x_std` - Evaluation point in [-1, 1]
/// 
/// # Returns
/// 
/// Value of Chebyshev series at x
fn chebyshev_evaluate_clenshaw(coefficients: &[f64], x_std: f64) -> f64 {
    let n = coefficients.len();
    
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coefficients[0];
    }
    
    // Clenshaw algorithm: backward recursion
    let mut b_k_plus_2 = 0.0;
    let mut b_k_plus_1 = 0.0;
    
    for k in (0..n).rev() {
        let b_k = coefficients[k] + 2.0 * x_std * b_k_plus_1 - b_k_plus_2;
        b_k_plus_2 = b_k_plus_1;
        b_k_plus_1 = b_k;
    }
    
    b_k_plus_1 - x_std * b_k_plus_2
}

/// Evaluate Chebyshev series using direct polynomial evaluation
/// 
/// Computes P(x) = Σₖ cₖTₖ(x) by explicitly evaluating each Tₖ.
/// 
/// # Arguments
/// 
/// * `coefficients` - Chebyshev coefficients
/// * `x_std` - Evaluation point in [-1, 1]
/// 
/// # Returns
/// 
/// Value of Chebyshev series at x
fn chebyshev_evaluate_direct(coefficients: &[f64], x_std: f64) -> f64 {
    let mut result = 0.0;
    
    for (k, &coef) in coefficients.iter().enumerate() {
        result += coef * chebyshev_polynomial(k, x_std);
    }
    
    result
}

/// Chebyshev Polynomial Interpolator
/// 
/// Interpolator using Chebyshev polynomial expansion with optimal nodes.
/// 
/// # Attributes
/// 
/// * `x_min`, `x_max` - Interval boundaries
/// * `nodes` - Pre-computed Chebyshev nodes
/// * `y_values` - Function values at nodes
/// * `coefficients` - Chebyshev expansion coefficients
/// * `n_points` - Number of interpolation points
/// * `use_clenshaw` - Whether to use Clenshaw algorithm
/// * `fitted` - Whether coefficients have been computed
#[gen_stub_pyclass]
#[pyclass]
pub struct ChebyshevInterpolator {
    x_min: f64,
    x_max: f64,
    nodes: Vec<f64>,
    y_values: Vec<f64>,
    coefficients: Vec<f64>,
    n_points: usize,
    use_clenshaw: bool,
    fitted: bool,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChebyshevInterpolator {
    /// Create a new Chebyshev interpolator
    /// 
    /// Parameters
    /// ----------
    /// n_points : int, default=10
    ///     Number of Chebyshev nodes (polynomial degree = n_points - 1)
    /// x_min : float, default=-1.0
    ///     Left endpoint of interval
    /// x_max : float, default=1.0
    ///     Right endpoint of interval
    /// use_clenshaw : bool, default=True
    ///     Whether to use Clenshaw algorithm (recommended)
    /// 
    /// Returns
    /// -------
    /// ChebyshevInterpolator
    ///     A new, unfitted interpolator instance
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If n_points is 0 or x_min >= x_max
    #[new]
    #[pyo3(signature = (n_points=10, x_min=-1.0, x_max=1.0, use_clenshaw=true))]
    pub fn new(n_points: usize, x_min: f64, x_max: f64, use_clenshaw: bool) -> PyResult<Self> {
        if n_points == 0 {
            return Err(PyValueError::new_err("n_points must be positive"));
        }
        if x_min >= x_max {
            return Err(PyValueError::new_err("x_min must be less than x_max"));
        }
        
        // Generate Chebyshev nodes
        let nodes = chebyshev_nodes(n_points, x_min, x_max);
        
        Ok(ChebyshevInterpolator {
            x_min,
            x_max,
            nodes,
            y_values: Vec::new(),
            coefficients: Vec::new(),
            n_points,
            use_clenshaw,
            fitted: false,
        })
    }

    /// Get the Chebyshev nodes
    /// 
    /// Returns
    /// -------
    /// list of float
    ///     Optimally-placed Chebyshev nodes in [x_min, x_max]
    /// 
    /// Notes
    /// -----
    /// These are the points where the function must be evaluated
    /// before calling fit().
    pub fn get_nodes(&self) -> Vec<f64> {
        self.nodes.clone()
    }

    /// Fit with function values at Chebyshev nodes
    /// 
    /// Parameters
    /// ----------
    /// y : list of float
    ///     Function values at the Chebyshev nodes
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If y length doesn't match number of nodes
    pub fn fit(&mut self, y: Vec<f64>) -> PyResult<()> {
        if y.len() != self.n_points {
            return Err(PyValueError::new_err(
                format!("Expected {} y values (one for each Chebyshev node), got {}", 
                        self.n_points, y.len())
            ));
        }
        
        self.y_values = y;
        self.coefficients = compute_chebyshev_coefficients(&self.y_values);
        self.fitted = true;
        Ok(())
    }

    /// Fit using a Python function
    /// 
    /// Convenience method that evaluates the function at Chebyshev nodes
    /// automatically.
    /// 
    /// Parameters
    /// ----------
    /// func : callable
    ///     Python function to interpolate
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If function call fails
    pub fn fit_function(&mut self, py: Python<'_>, func: Py<PyAny>) -> PyResult<()> {
        let mut y_values = Vec::with_capacity(self.n_points);
        
        for &x in &self.nodes {
            let result = func.call1(py, (x,))?;
            let y: f64 = result.extract(py)?;
            y_values.push(y);
        }
        
        self.fit(y_values)
    }

    /// Get Chebyshev polynomial coefficients
    /// 
    /// Returns
    /// -------
    /// list of float
    ///     Coefficients [c₀, c₁, ..., cₙ₋₁] for P(x) = Σ cₖTₖ(x)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If not fitted
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(y) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Set evaluation method
    /// 
    /// Parameters
    /// ----------
    /// use_clenshaw : bool
    ///     If True, use Clenshaw algorithm (stable, O(n))
    ///     If False, use direct evaluation (simple, O(n²))
    pub fn set_method(&mut self, use_clenshaw: bool) {
        self.use_clenshaw = use_clenshaw;
    }

    /// Evaluate the interpolation
    /// 
    /// Parameters
    /// ----------
    /// x : float or list of float
    ///     Point(s) at which to evaluate (must be in [x_min, x_max])
    /// 
    /// Returns
    /// -------
    /// float or list of float
    ///     Interpolated value(s)
    /// 
    /// Raises
    /// ------
    /// ValueError
    ///     If not fitted or x outside [x_min, x_max]
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(y) first."
            ));
        }

        if let Ok(single_x) = x.extract::<f64>() {
            if single_x < self.x_min || single_x > self.x_max {
                return Err(PyValueError::new_err(
                    format!("x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                            single_x, self.x_min, self.x_max)
                ));
            }
            
            let x_std = transform_to_standard(single_x, self.x_min, self.x_max);
            let result = if self.use_clenshaw {
                chebyshev_evaluate_clenshaw(&self.coefficients, x_std)
            } else {
                chebyshev_evaluate_direct(&self.coefficients, x_std)
            };
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Result<Vec<f64>, String> = x_list
                .iter()
                .map(|&xi| {
                    if xi < self.x_min || xi > self.x_max {
                        Err(format!("x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                                   xi, self.x_min, self.x_max))
                    } else {
                        let x_std = transform_to_standard(xi, self.x_min, self.x_max);
                        let result = if self.use_clenshaw {
                            chebyshev_evaluate_clenshaw(&self.coefficients, x_std)
                        } else {
                            chebyshev_evaluate_direct(&self.coefficients, x_std)
                        };
                        Ok(result)
                    }
                })
                .collect();
            
            match results {
                Ok(vals) => return Ok(vals.into_pyobject(py)?.into_any().unbind()),
                Err(e) => return Err(PyValueError::new_err(e)),
            }
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats"
        ))
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        let method = if self.use_clenshaw { "Clenshaw" } else { "Direct" };
        if self.fitted {
            format!(
                "ChebyshevInterpolator(n_points={}, x_range=[{:.2}, {:.2}], method={}, fitted)",
                self.n_points,
                self.x_min,
                self.x_max,
                method
            )
        } else {
            format!(
                "ChebyshevInterpolator(n_points={}, x_range=[{:.2}, {:.2}], method={}, not fitted)",
                self.n_points,
                self.x_min,
                self.x_max,
                method
            )
        }
    }
}
define_stub_info_gatherer!(stub_info);