use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Compute Hermite divided differences
/// This includes both function values and derivative values
fn hermite_divided_differences(
    x_values: &[f64],
    y_values: &[f64],
    dy_values: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = x_values.len();
    
    // Build z array (doubled x values) and q array (function and derivative values)
    let mut z = Vec::with_capacity(2 * n);
    let mut q = vec![vec![0.0; 2 * n]; 2 * n];
    
    // Initialize z and q[i][0]
    for i in 0..n {
        z.push(x_values[i]);
        z.push(x_values[i]);
        q[2 * i][0] = y_values[i];
        q[2 * i + 1][0] = y_values[i];
    }
    
    // First divided differences (derivatives)
    for i in 0..n {
        q[2 * i + 1][1] = dy_values[i];
        if i > 0 {
            q[2 * i][1] = (q[2 * i][0] - q[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1]);
        }
    }
    
    // Higher order divided differences
    for j in 2..2 * n {
        for i in j..2 * n {
            q[i][j] = (q[i][j - 1] - q[i - 1][j - 1]) / (z[i] - z[i - j]);
        }
    }
    
    // Extract coefficients (diagonal of q)
    let mut coefficients = Vec::with_capacity(2 * n);
    for i in 0..2 * n {
        coefficients.push(q[i][i]);
    }
    
    (z, coefficients)
}

/// Evaluate Hermite polynomial at a point
fn hermite_evaluate(z: &[f64], coefficients: &[f64], x: f64) -> f64 {
    let n = coefficients.len();
    if n == 0 {
        return f64::NAN;
    }
    
    let mut result = coefficients[n - 1];
    for i in (0..n - 1).rev() {
        result = result * (x - z[i]) + coefficients[i];
    }
    result
}

#[pyclass]
pub struct HermiteInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    dy_values: Vec<f64>,
    z_values: Vec<f64>,
    coefficients: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl HermiteInterpolator {
    #[new]
    pub fn new() -> Self {
        HermiteInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            dy_values: Vec::new(),
            z_values: Vec::new(),
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with x, y, and derivative (dy/dx) data points
    /// 
    /// Parameters:
    /// -----------
    /// x : list of float
    ///     x coordinates of data points
    /// y : list of float
    ///     y coordinates (function values) at data points
    /// dy : list of float
    ///     Derivatives (dy/dx) at data points
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>, dy: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() || x.len() != dy.len() {
            return Err(PyValueError::new_err(
                "x, y, and dy must all have the same length"
            ));
        }
        if x.is_empty() {
            return Err(PyValueError::new_err(
                "x, y, and dy cannot be empty"
            ));
        }
        
        self.x_values = x;
        self.y_values = y;
        self.dy_values = dy;
        
        // Compute Hermite coefficients
        let (z, coef) = hermite_divided_differences(
            &self.x_values,
            &self.y_values,
            &self.dy_values,
        );
        
        self.z_values = z;
        self.coefficients = coef;
        self.fitted = true;
        
        Ok(())
    }

    /// Get the Hermite polynomial coefficients
    pub fn get_coefficients(&self) -> PyResult<Vec<f64>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y, dy) first."
            ));
        }
        Ok(self.coefficients.clone())
    }

    /// Evaluate the interpolation at a single point or multiple points
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y, dy) first."
            ));
        }

        // Try to extract as a single float
        if let Ok(single_x) = x.extract::<f64>() {
            let result = hermite_evaluate(&self.z_values, &self.coefficients, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| hermite_evaluate(&self.z_values, &self.coefficients, xi))
                .collect();
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats"
        ))
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "HermiteInterpolator(fitted with {} points and derivatives)",
                self.x_values.len()
            )
        } else {
            "HermiteInterpolator(not fitted)".to_string()
        }
    }
}