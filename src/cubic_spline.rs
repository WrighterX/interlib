use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Cubic spline coefficients for one segment
#[derive(Clone, Debug)]
struct SplineSegment {
    a: f64, // Constant term
    b: f64, // Linear term
    c: f64, // Quadratic term
    d: f64, // Cubic term
    x: f64, // Left endpoint of segment
}

impl SplineSegment {
    /// Evaluate the cubic spline at point t (relative to segment start)
    fn eval(&self, x_val: f64) -> f64 {
        let dx = x_val - self.x;
        self.a + self.b * dx + self.c * dx * dx + self.d * dx * dx * dx
    }
}

/// Solve tridiagonal system using Thomas algorithm
/// For system: b[i]*x[i] + c[i]*x[i+1] = d[i] for i=0
///             a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i] for 0<i<n-1
///             a[i]*x[i-1] + b[i]*x[i] = d[i] for i=n-1
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];
    
    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    
    for i in 1..n {
        let m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
        c_prime[i] = c[i] * m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m;
    }
    
    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    
    x
}

/// Compute natural cubic spline coefficients
/// Natural boundary conditions: second derivative = 0 at endpoints
fn compute_natural_cubic_spline(x_values: &[f64], y_values: &[f64]) -> Vec<SplineSegment> {
    let n = x_values.len();
    
    if n < 2 {
        return vec![];
    }
    
    // Calculate h values (intervals)
    let mut h = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = x_values[i + 1] - x_values[i];
    }
    
    // Set up tridiagonal system for second derivatives
    // Natural spline: c[0] = c[n-1] = 0
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    
    // First row (natural boundary condition)
    b[0] = 1.0;
    c[0] = 0.0;
    d[0] = 0.0;
    
    // Interior rows
    for i in 1..n - 1 {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 3.0 * ((y_values[i + 1] - y_values[i]) / h[i] 
                    - (y_values[i] - y_values[i - 1]) / h[i - 1]);
    }
    
    // Last row (natural boundary condition)
    a[n - 1] = 0.0;
    b[n - 1] = 1.0;
    d[n - 1] = 0.0;
    
    // Solve for second derivatives (c coefficients in standard form)
    let c_coeffs = solve_tridiagonal(&a, &b, &c, &d);
    
    // Build spline segments
    let mut segments = Vec::new();
    
    for i in 0..n - 1 {
        let segment = SplineSegment {
            a: y_values[i],
            b: (y_values[i + 1] - y_values[i]) / h[i] 
                - h[i] * (c_coeffs[i + 1] + 2.0 * c_coeffs[i]) / 3.0,
            c: c_coeffs[i],
            d: (c_coeffs[i + 1] - c_coeffs[i]) / (3.0 * h[i]),
            x: x_values[i],
        };
        segments.push(segment);
    }
    
    segments
}

/// Evaluate cubic spline at a single point
fn cubic_spline_eval(segments: &[SplineSegment], x_values: &[f64], x: f64) -> f64 {
    if segments.is_empty() {
        return f64::NAN;
    }
    
    let n = x_values.len();
    
    // Handle boundary cases
    if x <= x_values[0] {
        return segments[0].eval(x);
    }
    if x >= x_values[n - 1] {
        return segments[segments.len() - 1].eval(x);
    }
    
    // Find the segment containing x
    for i in 0..segments.len() {
        if x >= x_values[i] && x <= x_values[i + 1] {
            return segments[i].eval(x);
        }
    }
    
    f64::NAN
}

#[pyclass]
pub struct CubicSplineInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    segments: Vec<SplineSegment>,
    fitted: bool,
}

#[pymethods]
impl CubicSplineInterpolator {
    #[new]
    pub fn new() -> Self {
        CubicSplineInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            segments: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with x and y data points
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() {
            return Err(PyValueError::new_err(
                "x and y must have the same length"
            ));
        }
        if x.len() < 2 {
            return Err(PyValueError::new_err(
                "Cubic spline interpolation requires at least 2 data points"
            ));
        }
        
        // Check if x values are sorted
        for i in 0..x.len() - 1 {
            if x[i] >= x[i + 1] {
                return Err(PyValueError::new_err(
                    "x values must be strictly increasing"
                ));
            }
        }
        
        self.x_values = x;
        self.y_values = y;
        
        // Compute spline coefficients
        self.segments = compute_natural_cubic_spline(&self.x_values, &self.y_values);
        
        if self.segments.is_empty() {
            return Err(PyValueError::new_err(
                "Failed to compute spline coefficients"
            ));
        }
        
        self.fitted = true;
        Ok(())
    }

    /// Get the number of spline segments
    pub fn num_segments(&self) -> PyResult<usize> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first."
            ));
        }
        Ok(self.segments.len())
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
            let result = cubic_spline_eval(&self.segments, &self.x_values, single_x);
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Try to extract as a list of floats
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| cubic_spline_eval(&self.segments, &self.x_values, xi))
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
                "CubicSplineInterpolator(fitted with {} points, {} segments, x range: [{:.2}, {:.2}])",
                self.x_values.len(),
                self.segments.len(),
                self.x_values.first().unwrap_or(&0.0),
                self.x_values.last().unwrap_or(&0.0)
            )
        } else {
            "CubicSplineInterpolator(not fitted)".to_string()
        }
    }
}