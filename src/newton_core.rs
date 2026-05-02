/// Newton Divided Differences Interpolation Core
///
/// This module implements the core logic for Newton's divided differences
/// interpolation, independent of any specific language interface.

#[derive(Clone, Debug)]
pub(crate) struct NewtonCore {
    x_values: Vec<f64>,
    coefficients: Vec<f64>,
    fitted: bool,
}

impl NewtonCore {
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
        if x.is_empty() {
            return Err("x and y cannot be empty".to_string());
        }

        // Divided differences turn the sample table into Newton coefficients.
        // Once this is computed, evaluation is just nested multiplication.
        self.coefficients = divided_differences(&x, &y);
        self.x_values = x;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        Ok(newton_evaluate(&self.x_values, &self.coefficients, x))
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0; xs.len()];
        self.fill_many(xs, &mut out)?;
        Ok(out)
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
            out[i] = newton_evaluate(&self.x_values, &self.coefficients, xs[i]);
            out[i + 1] = newton_evaluate(&self.x_values, &self.coefficients, xs[i + 1]);
            i += 2;
        }
        if i < xs.len() {
            out[i] = newton_evaluate(&self.x_values, &self.coefficients, xs[i]);
        }
        Ok(())
    }

    pub(crate) fn get_coefficients(&self) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        Ok(self.coefficients.clone())
    }

    pub(crate) fn repr(&self) -> String {
        if self.fitted {
            format!(
                "NewtonInterpolator(fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "NewtonInterpolator(not fitted)".to_string()
        }
    }
}

/// Compute divided differences table
fn divided_differences(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut coef = ys.to_vec();

    // Each outer pass raises the order of the difference table by one.
    // We update in place from the back so previously computed values stay valid.
    for j in 1..n {
        for i in (j..n).rev() {
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j]);
        }
    }
    coef
}

/// Evaluate Newton polynomial using Horner's method
#[inline]
fn newton_evaluate(xs: &[f64], coef: &[f64], x: f64) -> f64 {
    let n = coef.len();
    let mut result = coef[n - 1];

    // Horner-style evaluation of the Newton basis.
    for i in (0..n - 1).rev() {
        result = result * (x - xs[i]) + coef[i];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::NewtonCore;

    #[test]
    fn fit_and_evaluates() {
        let mut core = NewtonCore::new();
        core.fit(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();
        assert!((core.evaluate_single(0.5).unwrap() - 0.25).abs() < 1e-12);
        assert!((core.evaluate_single(1.5).unwrap() - 2.25).abs() < 1e-12);
    }

    #[test]
    fn evaluate_many_and_fill_many_match_scalar_path() {
        let mut core = NewtonCore::new();
        core.fit(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();

        let xs = vec![-1.0, 0.25, 1.25, 2.5];
        let scalar: Vec<f64> = xs
            .iter()
            .map(|&x| core.evaluate_single(x).unwrap())
            .collect();

        let many = core.evaluate_many(&xs).unwrap();
        assert_eq!(many, scalar);

        let mut out = vec![0.0; xs.len()];
        core.fill_many(&xs, &mut out).unwrap();
        assert_eq!(out, scalar);
    }
}
