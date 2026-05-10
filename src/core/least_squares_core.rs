use crate::core::core_error::CoreError;
use crate::core::core_trait::{InterpolationCore, solve_linear_system_gaussian};

fn compute_normal_equations(
    x_values: &[f64],
    y_values: &[f64],
    degree: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let n = x_values.len();
    let m = degree + 1;
    if n < m {
        return Err(format!("Need at least {} points for degree {}", m, degree));
    }

    let max_power = 2 * degree;
    let mut sum_xp = vec![0.0f64; max_power + 1];
    let mut sum_xyp = vec![0.0f64; m];

    for k in 0..n {
        let xk = x_values[k];
        let yk = y_values[k];
        let mut xp = 1.0;
        for p in 0..m {
            sum_xp[p] += xp;
            sum_xyp[p] += yk * xp;
            xp *= xk;
        }
        #[allow(clippy::needless_range_loop)]
        for p in m..=max_power {
            sum_xp[p] += xp;
            xp *= xk;
        }
    }

    let mut ata = vec![vec![0.0; m]; m];
    #[allow(clippy::needless_range_loop, clippy::manual_memcpy)]
    for i in 0..m {
        for j in 0..m {
            ata[i][j] = sum_xp[i + j];
        }
    }
    ata.push(sum_xyp);
    Ok(ata)
}

fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for &coef in coefficients.iter().rev() {
        result = result * x + coef;
    }
    result
}

fn evaluate_polynomial4(
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
) -> [f64; 4] {
    let mut r0 = 0.0;
    let mut r1 = 0.0;
    let mut r2 = 0.0;
    let mut r3 = 0.0;
    for &coef in coefficients.iter().rev() {
        r0 = r0 * x0 + coef;
        r1 = r1 * x1 + coef;
        r2 = r2 * x2 + coef;
        r3 = r3 * x3 + coef;
    }
    [r0, r1, r2, r3]
}

#[allow(clippy::too_many_arguments)]
fn evaluate_polynomial8(
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
    let mut r0 = 0.0;
    let mut r1 = 0.0;
    let mut r2 = 0.0;
    let mut r3 = 0.0;
    let mut r4 = 0.0;
    let mut r5 = 0.0;
    let mut r6 = 0.0;
    let mut r7 = 0.0;
    for &coef in coefficients.iter().rev() {
        r0 = r0 * x0 + coef;
        r1 = r1 * x1 + coef;
        r2 = r2 * x2 + coef;
        r3 = r3 * x3 + coef;
        r4 = r4 * x4 + coef;
        r5 = r5 * x5 + coef;
        r6 = r6 * x6 + coef;
        r7 = r7 * x7 + coef;
    }
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

#[derive(Clone)]
pub(crate) struct LeastSquaresCore {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    coefficients: Vec<f64>,
    degree: usize,
    fitted: bool,
}

impl LeastSquaresCore {
    pub(crate) fn new(degree: usize) -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
            coefficients: Vec::new(),
            degree,
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
        if x.len() <= self.degree {
            return Err(format!(
                "Need at least {} points for degree {} polynomial",
                self.degree + 1,
                self.degree
            ));
        }
        let ata = compute_normal_equations(&x, &y, self.degree)?;
        let (a, b) = ata.split_at(ata.len() - 1);
        let coeffs = solve_linear_system_gaussian(a.to_vec(), b[0].clone())?;
        self.x_values = x;
        self.y_values = y;
        self.coefficients = coeffs;
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

    pub(crate) fn degree(&self) -> usize {
        self.degree
    }

    pub(crate) fn r_squared(&self) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        let y_mean = self.y_values.iter().sum::<f64>() / self.y_values.len() as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for i in 0..self.x_values.len() {
            let y_pred = evaluate_polynomial(&self.coefficients, self.x_values[i]);
            ss_res += (self.y_values[i] - y_pred).powi(2);
            ss_tot += (self.y_values[i] - y_mean).powi(2);
        }
        if ss_tot == 0.0 {
            return Ok(1.0);
        }
        Ok(1.0 - ss_res / ss_tot)
    }

    pub(crate) fn repr(&self) -> String {
        if self.fitted {
            match self.r_squared() {
                Ok(r2) => format!(
                    "LeastSquaresInterpolator(degree={}, fitted with {} points, R²={:.4})",
                    self.degree,
                    self.x_values.len(),
                    r2
                ),
                Err(_) => format!(
                    "LeastSquaresInterpolator(degree={}, fitted with {} points)",
                    self.degree,
                    self.x_values.len()
                ),
            }
        } else {
            format!(
                "LeastSquaresInterpolator(degree={}, not fitted)",
                self.degree
            )
        }
    }
}

impl InterpolationCore for LeastSquaresCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(evaluate_polynomial(&self.coefficients, x))
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
            let values = evaluate_polynomial8(
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
            let values = evaluate_polynomial4(
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
            out[i] = evaluate_polynomial(&self.coefficients, xs[i]);
            i += 1;
        }
        Ok(())
    }
}
