use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// RBF kernel types
#[derive(Clone, Copy, Debug)]
pub enum RBFKernel {
    Gaussian,
    Multiquadric,
    InverseMultiquadric,
    ThinPlateSpline,
    Linear,
}

impl RBFKernel {
    /// Evaluate the RBF kernel at distance r
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

/// Compute RBF weights
fn compute_rbf_weights(
    x_values: &[f64],
    y_values: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> Result<Vec<f64>, String> {
    let n = x_values.len();
    
    // Build the RBF matrix
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

    /// Fit the RBF interpolator with x and y data points
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

    /// Get the RBF weights
    pub fn get_weights(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.weights.clone())
    }

    /// Evaluate the interpolation at a single point or multiple points
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, xi))
                .collect();
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats"
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