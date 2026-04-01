/// Cubic Spline Interpolation Core
///
/// This module implements the core logic for natural cubic spline interpolation,
/// independent of any specific language interface.

#[derive(Clone, Debug)]
pub(crate) struct SplineSegment {
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) c: f64,
    pub(crate) d: f64,
    pub(crate) x: f64,
}

impl SplineSegment {
    #[inline]
    pub(crate) fn eval(&self, x_val: f64) -> f64 {
        let dx = x_val - self.x;
        self.a + self.b * dx + self.c * dx * dx + self.d * dx * dx * dx
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CubicSplineCore {
    x_values: Vec<f64>,
    segments: Vec<SplineSegment>,
    fitted: bool,
}

impl CubicSplineCore {
    pub(crate) fn new() -> Self {
        Self {
            x_values: Vec::new(),
            segments: Vec::new(),
            fitted: false,
        }
    }

    pub(crate) fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("x and y must have the same length".to_string());
        }
        if x.len() < 2 {
            return Err("Cubic spline interpolation requires at least 2 data points".to_string());
        }

        for i in 0..x.len() - 1 {
            if x[i] >= x[i + 1] {
                return Err("x values must be strictly increasing".to_string());
            }
        }

        self.x_values = x;
        self.segments = compute_not_a_knot_spline(&self.x_values, &y);

        if self.segments.is_empty() {
            return Err("Failed to compute spline coefficients".to_string());
        }

        self.fitted = true;
        Ok(())
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        Ok(self.eval_internal(x))
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }

        let n = xs.len();
        let mut results = Vec::with_capacity(n);
        let mut i = 0;
        while i + 1 < n {
            results.push(self.eval_internal(xs[i]));
            results.push(self.eval_internal(xs[i + 1]));
            i += 2;
        }
        if i < n {
            results.push(self.eval_internal(xs[i]));
        }
        Ok(results)
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        if xs.len() != out.len() {
            return Err("input and output slices must have the same length".to_string());
        }

        let mut i = 0;
        while i + 1 < xs.len() {
            out[i] = self.eval_internal(xs[i]);
            out[i + 1] = self.eval_internal(xs[i + 1]);
            i += 2;
        }
        if i < xs.len() {
            out[i] = self.eval_internal(xs[i]);
        }
        Ok(())
    }

    #[inline]
    fn eval_internal(&self, val: f64) -> f64 {
        let n = self.x_values.len();
        if val <= self.x_values[0] {
            return self.segments[0].eval(val);
        }
        if val >= self.x_values[n - 1] {
            return self.segments[n - 2].eval(val);
        }

        let idx = match self
            .x_values
            .binary_search_by(|v| v.partial_cmp(&val).unwrap())
        {
            Ok(i) => {
                if i == n - 1 {
                    i - 1
                } else {
                    i
                }
            }
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };
        self.segments[idx].eval(val)
    }

    pub(crate) fn num_segments(&self) -> Result<usize, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        Ok(self.segments.len())
    }

    pub(crate) fn repr(&self) -> String {
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

fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
        if i < n - 1 {
            c_prime[i] = c[i] * m;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m;
    }

    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

fn compute_not_a_knot_spline(x: &[f64], y: &[f64]) -> Vec<SplineSegment> {
    let n = x.len();
    let mut h = vec![0.0; n - 1];
    let mut delta = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    let mut c_coeffs = vec![0.0; n];
    if n >= 4 {
        let m = n - 2;
        let mut a_sys = vec![0.0; m];
        let mut b_sys = vec![0.0; m];
        let mut c_sys = vec![0.0; m];
        let mut rhs = vec![0.0; m];

        for i in 1..m - 1 {
            let k = i + 1;
            a_sys[i] = h[k - 1];
            b_sys[i] = 2.0 * (h[k - 1] + h[k]);
            c_sys[i] = h[k];
            rhs[i] = 3.0 * (delta[k] - delta[k - 1]);
        }

        b_sys[0] = (3.0 * h[0] * h[1] + 2.0 * h[1] * h[1] + h[0] * h[0]) / h[1];
        c_sys[0] = (h[1] * h[1] - h[0] * h[0]) / h[1];
        rhs[0] = 3.0 * (delta[1] - delta[0]);

        let hn2 = h[n - 3];
        let hn1 = h[n - 2];
        a_sys[m - 1] = (hn2 * hn2 - hn1 * hn1) / hn2;
        b_sys[m - 1] = (3.0 * hn2 * hn1 + 2.0 * hn2 * hn2 + hn1 * hn1) / hn2;
        rhs[m - 1] = 3.0 * (delta[n - 2] - delta[n - 3]);

        let inner_c = solve_tridiagonal(&a_sys, &b_sys, &c_sys, &rhs);
        for i in 0..m {
            c_coeffs[i + 1] = inner_c[i];
        }
        c_coeffs[0] = ((h[0] + h[1]) * c_coeffs[1] - h[0] * c_coeffs[2]) / h[1];
        c_coeffs[n - 1] =
            ((h[n - 2] + h[n - 3]) * c_coeffs[n - 2] - h[n - 2] * c_coeffs[n - 3]) / h[n - 3];
    } else if n == 3 {
        // Fallback for n=3 (becomes a single quadratic if we had boundary conditions,
        // but not-a-knot usually needs 4 points. For 2-3 points we can use simple parabolas
        // or just linear. Here we'll just keep them zero for now as compute_not_a_knot_spline
        // traditionally expects n>=4. Let's provide a simple parabolic fit for n=3.)
        let h0 = x[1] - x[0];
        let h1 = x[2] - x[1];
        let d0 = (y[1] - y[0]) / h0;
        let d1 = (y[2] - y[1]) / h1;
        let c = (d1 - d0) / (h0 + h1);
        c_coeffs[0] = c;
        c_coeffs[1] = c;
        c_coeffs[2] = c;
    }

    let mut segments = Vec::new();
    for i in 0..n - 1 {
        segments.push(SplineSegment {
            a: y[i],
            b: delta[i] - h[i] * (2.0 * c_coeffs[i] + c_coeffs[i + 1]) / 3.0,
            c: c_coeffs[i],
            d: (c_coeffs[i + 1] - c_coeffs[i]) / (3.0 * h[i]),
            x: x[i],
        });
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::CubicSplineCore;

    #[test]
    fn fit_and_evaluates() {
        let mut core = CubicSplineCore::new();
        core.fit(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 4.0, 9.0, 16.0],
        )
        .unwrap();
        assert!((core.evaluate_single(0.5).unwrap() - 0.25).abs() < 1e-12);
        assert!((core.evaluate_single(2.5).unwrap() - 6.25).abs() < 1e-12);
    }
}
