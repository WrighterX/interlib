use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::f64::consts::PI;

/// Generate Chebyshev nodes of the first kind on interval [a, b]
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
fn transform_to_standard(x: f64, a: f64, b: f64) -> f64 {
    2.0 * (x - a) / (b - a) - 1.0
}

/// Compute Chebyshev polynomial T_n(x) using recurrence relation
fn chebyshev_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    
    let mut t_prev2 = 1.0; // T_0
    let mut t_prev1 = x;   // T_1
    let mut t_n = 0.0;
    
    for _ in 2..=n {
        t_n = 2.0 * x * t_prev1 - t_prev2;
        t_prev2 = t_prev1;
        t_prev1 = t_n;
    }
    
    t_n
}

/// Compute Chebyshev coefficients using DCT-like approach
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

/// Evaluate Chebyshev series at a point using Clenshaw algorithm
fn chebyshev_evaluate_clenshaw(coefficients: &[f64], x_std: f64) -> f64 {
    let n = coefficients.len();
    
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coefficients[0];
    }
    
    let mut b_k_plus_2 = 0.0;
    let mut b_k_plus_1 = 0.0;
    
    for k in (0..n).rev() {
        let b_k = coefficients[k] + 2.0 * x_std * b_k_plus_1 - b_k_plus_2;
        b_k_plus_2 = b_k_plus_1;
        b_k_plus_1 = b_k;
    }
    
    b_k_plus_1 - x_std * b_k_plus_2
}

/// Evaluate Chebyshev series using direct polynomial evaluation (alternative method)
fn chebyshev_evaluate_direct(coefficients: &[f64], x_std: f64) -> f64 {
    let mut result = 0.0;
    
    for (k, &coef) in coefficients.iter().enumerate() {
        result += coef * chebyshev_polynomial(k, x_std);
    }
    
    result
}

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

#[pymethods]
impl ChebyshevInterpolator {
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

    /// Get the Chebyshev nodes where the function should be evaluated
    pub fn get_nodes(&self) -> Vec<f64> {
        self.nodes.clone()
    }

    /// Fit the Chebyshev interpolator with function values at Chebyshev nodes
    /// 
    /// Parameters:
    /// -----------
    /// y : list of float
    ///     Function values at the Chebyshev nodes (must match number of nodes)
    pub fn fit(&mut self, y: Vec<f64>) -> PyResult<()> {
        if y.len() != self.n_points {
            return Err(PyValueError::new_err(
                format!("Expected {} y values (one for each Chebyshev node), got {}", 
                        self.n_points, y.len())
            ));
        }
        
        self.y_values = y;
        
        // Compute Chebyshev coefficients
        self.coefficients = compute_chebyshev_coefficients(&self.y_values);
        
        self.fitted = true;
        Ok(())
    }

    /// Fit using a function directly (evaluates at Chebyshev nodes automatically)
    /// This is a convenience method for Python users
    pub fn fit_function(&mut self, py: Python<'_>, func: Py<PyAny>) -> PyResult<()> {
        let mut y_values = Vec::with_capacity(self.n_points);
        
        for &x in &self.nodes {
            let result = func.call1(py, (x,))?;
            let y: f64 = result.extract(py)?;
            y_values.push(y);
        }
        
        self.fit(y_values)
    }

    /// Get the Chebyshev polynomial coefficients
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(y) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Set evaluation method (Clenshaw vs direct polynomial)
    pub fn set_method(&mut self, use_clenshaw: bool) {
        self.use_clenshaw = use_clenshaw;
    }

    /// Evaluate the interpolation at a single point or multiple points
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(y) first."
            ));
        }

        // Try to extract as a single float
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

        // Try to extract as a list of floats
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