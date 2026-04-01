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
        for p in m..=max_power {
            sum_xp[p] += xp;
            xp *= xk;
        }
    }

    let mut ata = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            ata[i][j] = sum_xp[i + j];
        }
    }
    let atb = sum_xyp;
    ata.push(atb);
    Ok(ata)
}

fn solve_linear_system(ata: Vec<Vec<f64>>) -> Result<Vec<f64>, String> {
    let n = ata.len() - 1;
    let mut a = ata[..n].to_vec();
    let mut b = ata[n].clone();

    for k in 0..n {
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();
        for i in k + 1..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_idx = i;
            }
        }
        if max_idx != k {
            a.swap(k, max_idx);
            b.swap(k, max_idx);
        }
        if a[k][k].abs() < 1e-12 {
            return Err("Singular matrix".into());
        }
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    Ok(x)
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
            return Err("x and y must have the same length".to_string());
        }
        if x.is_empty() {
            return Err("x and y cannot be empty".to_string());
        }
        if x.len() <= self.degree {
            return Err(format!(
                "Need at least {} points for degree {} polynomial",
                self.degree + 1,
                self.degree
            ));
        }
        let ata = compute_normal_equations(&x, &y, self.degree)?;
        let coeffs = solve_linear_system(ata)?;
        self.x_values = x;
        self.y_values = y;
        self.coefficients = coeffs;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn get_coefficients(&self) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("Interpolator not fitted".to_string());
        }
        Ok(self.coefficients.clone())
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted".to_string());
        }
        Ok(evaluate_polynomial(&self.coefficients, x))
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted".to_string());
        }
        if xs.len() != out.len() {
            return Err("Input/output length mismatch".to_string());
        }
        let mut i = 0;
        while i + 1 < xs.len() {
            out[i] = evaluate_polynomial(&self.coefficients, xs[i]);
            out[i + 1] = evaluate_polynomial(&self.coefficients, xs[i + 1]);
            i += 2;
        }
        if i < xs.len() {
            out[i] = evaluate_polynomial(&self.coefficients, xs[i]);
        }
        Ok(())
    }

    pub(crate) fn r_squared(&self) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted".to_string());
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
            format!("LeastSquaresInterpolator(degree={}, not fitted)", self.degree)
        }
    }
}
