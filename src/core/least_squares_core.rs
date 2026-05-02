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

    let max_power = 2 * m - 1;
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
    coefficients
        .iter()
        .enumerate()
        .map(|(power, coef)| coef * x.powi(power as i32))
        .sum()
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
        while i + 1 < n {
            out[i] = evaluate_polynomial(&self.coefficients, xs[i]);
            out[i + 1] =
                evaluate_polynomial(&self.coefficients, xs[i + 1]);
            i += 2;
        }
        if i < n {
            out[i] = evaluate_polynomial(&self.coefficients, xs[i]);
        }
        Ok(())
    }
}
