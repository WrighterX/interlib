/// Piecewise Quadratic Interpolation Core
///
/// This module implements the core logic for piecewise quadratic interpolation,
/// independent of any specific language interface.

#[derive(Clone, Debug)]
pub(crate) struct QuadraticCore {
    x_values: Vec<f64>,
    coefficients: Vec<f64>, // flat: [a0, b0, c0, a1, b1, c1, ...]
    fitted: bool,
}

impl QuadraticCore {
    pub(crate) fn new() -> Self {
        Self {
            x_values: Vec::new(),
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    pub(crate) fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("x and y must have the same length".to_string());
        }
        if x.len() < 3 {
            return Err("Quadratic interpolation requires at least 3 data points".to_string());
        }

        // Check if x values are strictly increasing
        for i in 0..x.len() - 1 {
            if x[i] >= x[i + 1] {
                return Err("x values must be strictly increasing".to_string());
            }
        }

        let n = x.len();
        let n_segments = n - 2; // overlapping triplets: [0,1,2], [1,2,3], ..., [n-3,n-2,n-1]

        // Pre-compute all (a, b, c) coefficients — flat layout for cache locality
        let mut coefficients = Vec::with_capacity(n_segments * 3);
        for i in 0..n_segments {
            let (a, b, c) = solve_quadratic_coefficients(
                x[i], y[i],
                x[i + 1], y[i + 1],
                x[i + 2], y[i + 2],
            );
            coefficients.push(a);
            coefficients.push(b);
            coefficients.push(c);
        }

        self.x_values = x;
        self.coefficients = coefficients;
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
    fn eval_internal(&self, x: f64) -> f64 {
        let x_values = &self.x_values;
        let n = x_values.len();
        let n_segments = n - 2;

        // Binary search: find first index where x_values[i] > x
        let pos = x_values.partition_point(|&xi| xi <= x);

        // Map position to segment index, clamped to valid range [0, n_segments - 1]
        let seg_idx = pos.saturating_sub(2).min(n_segments - 1);

        let base = seg_idx * 3;
        let a = self.coefficients[base];
        let b = self.coefficients[base + 1];
        let c = self.coefficients[base + 2];

        eval_quadratic(a, b, c, x)
    }

    pub(crate) fn repr(&self) -> String {
        if self.fitted {
            format!(
                "QuadraticInterpolator(fitted with {} points, x range: [{:.2}, {:.2}])",
                self.x_values.len(),
                self.x_values.first().unwrap_or(&0.0),
                self.x_values.last().unwrap_or(&0.0)
            )
        } else {
            "QuadraticInterpolator(not fitted)".to_string()
        }
    }
}

/// Solve 3×3 system for quadratic coefficients using Cramer's rule
#[inline]
fn solve_quadratic_coefficients(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (f64, f64, f64) {
    let x0_sq = x0 * x0;
    let x1_sq = x1 * x1;
    let x2_sq = x2 * x2;

    let det = 1.0 * (x1 * x2_sq - x2 * x1_sq)
            - x0 * (1.0 * x2_sq - 1.0 * x1_sq)
            + x0_sq * (1.0 * x2 - 1.0 * x1);

    if det.abs() < 1e-12 {
        let slope = (y1 - y0) / (x1 - x0);
        let intercept = y0 - slope * x0;
        return (intercept, slope, 0.0);
    }

    let da = y0 * (x1 * x2_sq - x2 * x1_sq) - y1 * (x0 * x2_sq - x2 * x0_sq) + y2 * (x0 * x1_sq - x1 * x0_sq);
    let db = 1.0 * (y1 * x2_sq - y2 * x1_sq) - 1.0 * (y0 * x2_sq - y2 * x0_sq) + 1.0 * (y0 * x1_sq - y1 * x0_sq);
    let dc = 1.0 * (x1 * y2 - x2 * y1)
           - x0 * (1.0 * y2 - 1.0 * y1)
           + y0 * (1.0 * x2 - 1.0 * x1);

    (da / det, db / det, dc / det)
}

/// Evaluate quadratic polynomial at a point
#[inline]
fn eval_quadratic(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a + b * x + c * x * x
}

#[cfg(test)]
mod tests {
    use super::QuadraticCore;

    #[test]
    fn fit_and_evaluates() {
        let mut core = QuadraticCore::new();
        // y = x^2
        core.fit(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 4.0, 9.0]).unwrap();
        assert!((core.evaluate_single(0.5).unwrap() - 0.25).abs() < 1e-12);
        assert!((core.evaluate_single(1.5).unwrap() - 2.25).abs() < 1e-12);
        assert!((core.evaluate_single(2.5).unwrap() - 6.25).abs() < 1e-12);
    }
}
