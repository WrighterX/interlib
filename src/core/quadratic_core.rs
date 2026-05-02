use crate::core::core_error::CoreError;
use crate::core::core_trait::InterpolationCore;

#[derive(Clone, Debug)]
pub(crate) struct QuadraticCore {
    x_values: Vec<f64>,
    coefficients: Vec<f64>,
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
            return Err(CoreError::LengthMismatch {
                left_name: "x",
                left: x.len(),
                right_name: "y",
                right: y.len(),
            }
            .into());
        }
        if x.len() < 3 {
            return Err("Quadratic interpolation requires at least 3 data points".to_string());
        }

        for i in 0..x.len() - 1 {
            if x[i] >= x[i + 1] {
                return Err("x values must be strictly increasing".to_string());
            }
        }

        let n = x.len();
        let n_segments = n - 2;

        let mut coefficients = Vec::with_capacity(n_segments * 3);
        for i in 0..n_segments {
            let (a, b, c) =
                solve_quadratic_coefficients(x[i], y[i], x[i + 1], y[i + 1], x[i + 2], y[i + 2]);
            coefficients.push(a);
            coefficients.push(b);
            coefficients.push(c);
        }

        self.x_values = x;
        self.coefficients = coefficients;
        self.fitted = true;
        Ok(())
    }

    fn fill_many_sorted(&self, xs: &[f64], out: &mut [f64]) {
        let x_values = &self.x_values;
        let n = x_values.len();
        let n_segments = n - 2;
        let mut pos = 0usize;

        for (i, &x) in xs.iter().enumerate() {
            while pos < n && x_values[pos] <= x {
                pos += 1;
            }

            let seg_idx = pos.saturating_sub(2).min(n_segments - 1);
            let base = seg_idx * 3;
            let a = self.coefficients[base];
            let b = self.coefficients[base + 1];
            let c = self.coefficients[base + 2];
            out[i] = eval_quadratic(a, b, c, x);
        }
    }

    #[inline]
    pub(crate) fn eval_internal(&self, x: f64) -> f64 {
        let x_values = &self.x_values;
        let n = x_values.len();
        let n_segments = n - 2;

        let pos = x_values.partition_point(|&xi| xi <= x);

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

impl InterpolationCore for QuadraticCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(self.eval_internal(x))
    }

    fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        if xs.len() != out.len() {
            return Err(CoreError::LengthMismatch {
                left_name: "input",
                left: xs.len(),
                right_name: "output",
                right: out.len(),
            }
            .into());
        }

        if crate::core::core_trait::is_non_decreasing(xs) {
            self.fill_many_sorted(xs, out);
            return Ok(());
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
}

/// Solve 3×3 system for quadratic coefficients using Cramer's rule
#[inline]
fn solve_quadratic_coefficients(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> (f64, f64, f64) {
    let x0_sq = x0 * x0;
    let x1_sq = x1 * x1;
    let x2_sq = x2 * x2;

    // Solve the 3x3 system analytically; the fallback below handles the
    // degenerate case where the local points are effectively collinear.
    let det = 1.0 * (x1 * x2_sq - x2 * x1_sq) - x0 * (1.0 * x2_sq - 1.0 * x1_sq)
        + x0_sq * (1.0 * x2 - 1.0 * x1);

    if det.abs() < 1e-12 {
        let slope = (y1 - y0) / (x1 - x0);
        let intercept = y0 - slope * x0;
        return (intercept, slope, 0.0);
    }

    let da = y0 * (x1 * x2_sq - x2 * x1_sq) - y1 * (x0 * x2_sq - x2 * x0_sq)
        + y2 * (x0 * x1_sq - x1 * x0_sq);
    let db = 1.0 * (y1 * x2_sq - y2 * x1_sq) - 1.0 * (y0 * x2_sq - y2 * x0_sq)
        + 1.0 * (y0 * x1_sq - y1 * x0_sq);
    let dc = 1.0 * (x1 * y2 - x2 * y1) - x0 * (1.0 * y2 - 1.0 * y1) + y0 * (1.0 * x2 - 1.0 * x1);

    (da / det, db / det, dc / det)
}

#[inline]
fn eval_quadratic(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a + b * x + c * x * x
}

#[cfg(test)]
mod tests {
    use super::QuadraticCore;
    use crate::core::core_trait::InterpolationCore;

    #[test]
    fn fit_and_evaluates() {
        let mut core = QuadraticCore::new();
        // y = x^2
        core.fit(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 4.0, 9.0])
            .unwrap();
        assert!((core.evaluate_single(0.5).unwrap() - 0.25).abs() < 1e-12);
        assert!((core.evaluate_single(1.5).unwrap() - 2.25).abs() < 1e-12);
        assert!((core.evaluate_single(2.5).unwrap() - 6.25).abs() < 1e-12);
    }

    #[test]
    fn evaluate_many_and_fill_many_match_scalar_path() {
        let mut core = QuadraticCore::new();
        core.fit(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 4.0, 9.0])
            .unwrap();

        let xs = vec![-0.5, 0.5, 1.5, 2.5, 3.5];
        let scalar: Vec<f64> = xs
            .iter()
            .map(|&x| core.evaluate_single(x).unwrap())
            .collect();

        let many = core.evaluate_many(&xs).unwrap();
        assert_eq!(many, scalar);

        let mut out = vec![0.0; xs.len()];
        core.fill_many(&xs, &mut out).unwrap();
        assert_eq!(out, scalar);

        let xs_unsorted = vec![2.5, -0.5, 3.5, 1.5, 0.5];
        let scalar_unsorted: Vec<f64> = xs_unsorted
            .iter()
            .map(|&x| core.evaluate_single(x).unwrap())
            .collect();
        let mut out_unsorted = vec![0.0; xs_unsorted.len()];
        core.fill_many(&xs_unsorted, &mut out_unsorted).unwrap();
        assert_eq!(out_unsorted, scalar_unsorted);
    }
}
