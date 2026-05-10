use crate::core::core_error::CoreError;
use crate::core::core_trait::InterpolationCore;

#[derive(Clone)]
pub(crate) struct HermiteCore {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    dy_values: Vec<f64>,
    z_values: Vec<f64>,
    coefficients: Vec<f64>,
    fitted: bool,
}

impl HermiteCore {
    pub(crate) fn new() -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
            dy_values: Vec::new(),
            z_values: Vec::new(),
            coefficients: Vec::new(),
            fitted: false,
        }
    }

    pub(crate) fn fit(&mut self, x: Vec<f64>, y: Vec<f64>, dy: Vec<f64>) -> Result<(), String> {
        if x.len() != y.len() || x.len() != dy.len() {
            return Err(CoreError::LengthMismatch {
                left_name: "x",
                left: x.len(),
                right_name: "y/dy",
                right: y.len().max(dy.len()),
            }
            .into());
        }
        if x.is_empty() {
            return Err(CoreError::EmptyInput {
                what: "x, y, and dy",
            }
            .into());
        }

        self.x_values = x;
        self.y_values = y;
        self.dy_values = dy;

        let (z, coef) =
            hermite_divided_differences(&self.x_values, &self.y_values, &self.dy_values);

        self.z_values = z;
        self.coefficients = coef;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn get_coefficients(&self) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y, dy) first.",
            }
            .into());
        }
        Ok(self.coefficients.clone())
    }

    pub(crate) fn repr(&self) -> String {
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

impl InterpolationCore for HermiteCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y, dy) first.",
            }
            .into());
        }
        Ok(hermite_evaluate(&self.z_values, &self.coefficients, x))
    }

    fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y, dy) first.",
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
            let values = hermite_evaluate8(
                &self.z_values,
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
            let values = hermite_evaluate4(
                &self.z_values,
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
            out[i] = hermite_evaluate(&self.z_values, &self.coefficients, xs[i]);
            i += 1;
        }
        Ok(())
    }
}

fn hermite_divided_differences(
    x_values: &[f64],
    y_values: &[f64],
    dy_values: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = x_values.len();
    let m = 2 * n;
    let mut z = Vec::with_capacity(m);
    let mut q = vec![0.0; m * m];
    let stride = m;

    // Each sample point is repeated twice in the augmented node vector.
    for i in 0..n {
        z.push(x_values[i]);
        z.push(x_values[i]);
        q[2 * i * stride] = y_values[i];
        q[(2 * i + 1) * stride] = y_values[i];
    }

    for i in 0..n {
        // The first off-diagonal stores derivatives directly when nodes repeat.
        q[(2 * i + 1) * stride + 1] = dy_values[i];
        if i > 0 {
            let idx_cur = 2 * i * stride;
            let idx_prev = (2 * i - 1) * stride;
            let denominator = z[2 * i] - z[2 * i - 1];
            q[2 * i * stride + 1] = (q[idx_cur] - q[idx_prev]) / denominator;
        }
    }

    // Higher-order divided differences are filled in place using the usual
    // Newton-table recurrence.
    for j in 2..m {
        let mut i = j;
        while i + 1 < m {
            let z_i_0 = z[i];
            let z_ij_0 = z[i - j];
            let denominator_0 = z_i_0 - z_ij_0;
            let q_cur_prev_0 = q[i * stride + j - 1];
            let q_prev_row_0 = q[(i - 1) * stride + j - 1];
            q[i * stride + j] = (q_cur_prev_0 - q_prev_row_0) / denominator_0;

            let z_i_1 = z[i + 1];
            let z_ij_1 = z[i + 1 - j];
            let denominator_1 = z_i_1 - z_ij_1;
            let q_cur_prev_1 = q[(i + 1) * stride + j - 1];
            let q_prev_row_1 = q[i * stride + j - 1];
            q[(i + 1) * stride + j] = (q_cur_prev_1 - q_prev_row_1) / denominator_1;

            i += 2;
        }
        if i < m {
            let z_i = z[i];
            let z_ij = z[i - j];
            let denominator = z_i - z_ij;
            let q_cur_prev = q[i * stride + j - 1];
            let q_prev_row = q[(i - 1) * stride + j - 1];
            q[i * stride + j] = (q_cur_prev - q_prev_row) / denominator;
        }
    }

    let mut coefficients = Vec::with_capacity(m);
    for i in 0..m {
        coefficients.push(q[i * stride + i]);
    }
    (z, coefficients)
}

#[inline]
fn hermite_evaluate(z: &[f64], coefficients: &[f64], x: f64) -> f64 {
    let n = coefficients.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut result = coefficients[n - 1];
    // Horner evaluation on the Hermite/Newton basis.
    for i in (0..n - 1).rev() {
        result = result * (x - z[i]) + coefficients[i];
    }
    result
}

fn hermite_evaluate4(
    z: &[f64],
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
) -> [f64; 4] {
    let n = coefficients.len();
    if n == 0 {
        return [f64::NAN; 4];
    }
    let mut r0 = coefficients[n - 1];
    let mut r1 = r0;
    let mut r2 = r0;
    let mut r3 = r0;

    for i in (0..n - 1).rev() {
        let node = z[i];
        let c = coefficients[i];
        r0 = r0 * (x0 - node) + c;
        r1 = r1 * (x1 - node) + c;
        r2 = r2 * (x2 - node) + c;
        r3 = r3 * (x3 - node) + c;
    }

    [r0, r1, r2, r3]
}

#[allow(clippy::too_many_arguments)]
fn hermite_evaluate8(
    z: &[f64],
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    x5: f64,
    x6: f64,
    x7: f64,
) -> [f64; 8] {
    let n = coefficients.len();
    if n == 0 {
        return [f64::NAN; 8];
    }
    let mut r0 = coefficients[n - 1];
    let mut r1 = r0;
    let mut r2 = r0;
    let mut r3 = r0;
    let mut r4 = r0;
    let mut r5 = r0;
    let mut r6 = r0;
    let mut r7 = r0;

    for i in (0..n - 1).rev() {
        let node = z[i];
        let c = coefficients[i];
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
    use super::HermiteCore;
    use crate::core::core_trait::InterpolationCore;

    #[test]
    fn fit_and_evaluates() {
        let mut core = HermiteCore::new();
        core.fit(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 4.0],
            vec![0.0, 2.0, 4.0],
        )
        .unwrap();

        let v = core.evaluate_single(1.5).unwrap();
        assert!(v.is_finite());
    }

    #[test]
    fn evaluate_many_and_fill_many_match_scalar_path() {
        let mut core = HermiteCore::new();
        core.fit(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 4.0],
            vec![0.0, 2.0, 4.0],
        )
        .unwrap();

        let xs = vec![0.0, 0.5, 1.5, 2.0];
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
