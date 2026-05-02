/// Chebyshev Polynomial Interpolation support shared between Python and FFI.
///
/// This module retains the mathematical helper functions and exposes a `ChebyshevCore`
/// that can be used by the Python bindings, the FFI exports, or other runtimes.
use std::f64::consts::PI;

/// Generate Chebyshev nodes of the first kind on interval [a, b].
fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    let mut nodes = Vec::with_capacity(n);
    for k in 0..n {
        // First compute the node on [-1, 1], then map it into [a, b].
        let x = ((2 * k + 1) as f64 * PI / (2 * n) as f64).cos();
        let x_transformed = 0.5 * (b - a) * x + 0.5 * (b + a);
        nodes.push(x_transformed);
    }
    nodes
}

/// Linear transform from [a, b] to [-1, 1].
fn transform_to_standard(x: f64, a: f64, b: f64) -> f64 {
    // All coefficient math happens on the standard Chebyshev interval.
    2.0 * (x - a) / (b - a) - 1.0
}

/// Evaluate Chebyshev polynomial T_n(x) using recurrence.
fn chebyshev_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut t_n_minus_two = 1.0;
    let mut t_n_minus_one = x;
    let mut t_n = x;
    for _ in 2..=n {
        t_n = 2.0 * x * t_n_minus_one - t_n_minus_two;
        t_n_minus_two = t_n_minus_one;
        t_n_minus_one = t_n;
    }
    t_n
}

/// Compute Chebyshev coefficients using discrete cosine transform.
fn compute_chebyshev_coefficients(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut coefficients = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            // This is the discrete cosine transform form of the Chebyshev fit.
            let angle = (k * (2 * j + 1)) as f64 * PI / (2 * n) as f64;
            sum += y[j] * (angle).cos();
        }
        let value = if k == 0 {
            sum / n as f64
        } else {
            (2.0 * sum) / n as f64
        };
        coefficients.push(value);
    }
    coefficients
}

/// Clenshaw evaluation for stability.
fn chebyshev_evaluate_clenshaw(coefficients: &[f64], x_std: f64) -> f64 {
    let n = coefficients.len();
    let mut b_k_plus_two = 0.0;
    let mut b_k_plus_one = 0.0;
    for k in (0..n).rev() {
        // Backward recurrence keeps the evaluation numerically stable.
        let b_k = coefficients[k] + 2.0 * x_std * b_k_plus_one - b_k_plus_two;
        b_k_plus_two = b_k_plus_one;
        b_k_plus_one = b_k;
    }
    b_k_plus_one - x_std * b_k_plus_two
}

/// Direct polynomial evaluation.
fn chebyshev_evaluate_direct(coefficients: &[f64], x_std: f64) -> f64 {
    let mut result = 0.0;
    for (k, &coef) in coefficients.iter().enumerate() {
        // The direct path is simpler, but it is less stable than Clenshaw.
        result += coef * chebyshev_polynomial(k, x_std);
    }
    result
}

/// Shared Chebyshev core implementation.
pub struct ChebyshevCore {
    x_min: f64,
    x_max: f64,
    nodes: Vec<f64>,
    coefficients: Vec<f64>,
    n_points: usize,
    use_clenshaw: bool,
    fitted: bool,
}

impl ChebyshevCore {
    pub(crate) fn new(
        n_points: usize,
        x_min: f64,
        x_max: f64,
        use_clenshaw: bool,
    ) -> Result<Self, String> {
        if n_points == 0 {
            return Err("n_points must be positive".into());
        }
        if x_min >= x_max {
            return Err("x_min must be less than x_max".into());
        }
        let nodes = chebyshev_nodes(n_points, x_min, x_max);
        Ok(Self {
            x_min,
            x_max,
            nodes,
            coefficients: Vec::new(),
            n_points,
            use_clenshaw,
            fitted: false,
        })
    }

    pub(crate) fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    pub(crate) fn n_points(&self) -> usize {
        self.n_points
    }

    pub(crate) fn fit(&mut self, y: &[f64]) -> Result<(), String> {
        if y.len() != self.n_points {
            return Err(format!(
                "Expected {} y values (one for each Chebyshev node), got {}",
                self.n_points,
                y.len()
            ));
        }
        // The fit is a transform from sampled values to Chebyshev coefficients.
        self.coefficients = compute_chebyshev_coefficients(y);
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn coefficients(&self) -> Result<&[f64], String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(y) first.".into());
        }
        Ok(&self.coefficients)
    }

    pub(crate) fn set_method(&mut self, use_clenshaw: bool) {
        self.use_clenshaw = use_clenshaw;
    }

    fn ensure_in_range(&self, x: f64) -> Result<f64, String> {
        if x < self.x_min || x > self.x_max {
            return Err(format!(
                "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                x, self.x_min, self.x_max
            ));
        }
        Ok(transform_to_standard(x, self.x_min, self.x_max))
    }

    fn evaluate_impl(&self, x_std: f64) -> f64 {
        if self.use_clenshaw {
            chebyshev_evaluate_clenshaw(&self.coefficients, x_std)
        } else {
            chebyshev_evaluate_direct(&self.coefficients, x_std)
        }
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(y) first.".into());
        }
        let x_std = self.ensure_in_range(x)?;
        Ok(self.evaluate_impl(x_std))
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if xs.len() != out.len() {
            return Err("Input/output length mismatch".into());
        }
        for (i, &value) in xs.iter().enumerate() {
            // MATLAB and FFI use this path for batch evaluation.
            let x_std = self.ensure_in_range(value)?;
            out[i] = self.evaluate_impl(x_std);
        }
        Ok(())
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0; xs.len()];
        self.fill_many(xs, &mut out)?;
        Ok(out)
    }

    pub(crate) fn repr(&self) -> String {
        let method = if self.use_clenshaw {
            "Clenshaw"
        } else {
            "Direct"
        };
        if self.fitted {
            format!(
                "ChebyshevInterpolator(n_points={}, x_range=[{:.2}, {:.2}], method={}, fitted)",
                self.n_points, self.x_min, self.x_max, method
            )
        } else {
            format!(
                "ChebyshevInterpolator(n_points={}, x_range=[{:.2}, {:.2}], method={}, not fitted)",
                self.n_points, self.x_min, self.x_max, method
            )
        }
    }
}
