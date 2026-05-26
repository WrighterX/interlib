use crate::core::core_error::CoreError;
use crate::core::core_trait::InterpolationCore;

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
            return Err(CoreError::LengthMismatch {
                left_name: "x",
                left: x.len(),
                right_name: "y",
                right: y.len(),
            }
            .into());
        }
        if x.is_empty() {
            return Err(CoreError::EmptyInput { what: "x and y" }.into());
        }

        self.coefficients = divided_differences(&x, &y);
        self.x_values = x;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn get_coefficients(&self) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
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

impl InterpolationCore for NewtonCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(newton_evaluate(&self.x_values, &self.coefficients, x))
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
        let n = xs.len();
        let mut i = 0;
        while i + 7 < n {
            let values = newton_evaluate8(
                &self.x_values,
                &self.coefficients,
                xs[i],
                xs[i + 1],
                xs[i + 2],
                xs[i + 3],
                xs[i + 4],
                xs[i + 5],
                xs[i + 6],
                xs[i + 7],
            );
            out[i] = values[0];
            out[i + 1] = values[1];
            out[i + 2] = values[2];
            out[i + 3] = values[3];
            out[i + 4] = values[4];
            out[i + 5] = values[5];
            out[i + 6] = values[6];
            out[i + 7] = values[7];
            i += 8;
        }
        while i + 3 < n {
            let values = newton_evaluate4(
                &self.x_values,
                &self.coefficients,
                xs[i],
                xs[i + 1],
                xs[i + 2],
                xs[i + 3],
            );
            out[i] = values[0];
            out[i + 1] = values[1];
            out[i + 2] = values[2];
            out[i + 3] = values[3];
            i += 4;
        }
        while i < n {
            out[i] = newton_evaluate(&self.x_values, &self.coefficients, xs[i]);
            i += 1;
        }
        Ok(())
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

fn newton_evaluate4(xs: &[f64], coef: &[f64], x0: f64, x1: f64, x2: f64, x3: f64) -> [f64; 4] {
    let n = coef.len();
    let mut r0 = coef[n - 1];
    let mut r1 = r0;
    let mut r2 = r0;
    let mut r3 = r0;

    for i in (0..n - 1).rev() {
        let node = xs[i];
        let c = coef[i];
        r0 = r0 * (x0 - node) + c;
        r1 = r1 * (x1 - node) + c;
        r2 = r2 * (x2 - node) + c;
        r3 = r3 * (x3 - node) + c;
    }

    [r0, r1, r2, r3]
}

#[allow(clippy::too_many_arguments)]
fn newton_evaluate8(
    xs: &[f64],
    coef: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    x5: f64,
    x6: f64,
    x7: f64,
) -> [f64; 8] {
    let n = coef.len();
    let mut r0 = coef[n - 1];
    let mut r1 = r0;
    let mut r2 = r0;
    let mut r3 = r0;
    let mut r4 = r0;
    let mut r5 = r0;
    let mut r6 = r0;
    let mut r7 = r0;

    for i in (0..n - 1).rev() {
        let node = xs[i];
        let c = coef[i];
        r0 = r0 * (x0 - node) + c;
        r1 = r1 * (x1 - node) + c;
        r2 = r2 * (x2 - node) + c;
        r3 = r3 * (x3 - node) + c;
        r4 = r4 * (x4 - node) + c;
        r5 = r5 * (x5 - node) + c;
        r6 = r6 * (x6 - node) + c;
        r7 = r7 * (x7 - node) + c;
    }

    [r0, r1, r2, r3, r4, r5, r6, r7]
}

#[cfg(test)]
mod tests {
    use super::NewtonCore;
    use crate::core::core_trait::InterpolationCore;

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
