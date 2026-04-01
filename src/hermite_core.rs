/// Hermite interpolation core shared between Python and C ABI fronts.
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
            return Err("x, y, and dy must all have the same length".to_string());
        }
        if x.is_empty() {
            return Err("x, y, and dy cannot be empty".to_string());
        }

        self.x_values = x;
        self.y_values = y;
        self.dy_values = dy;

        let (z, coef) = hermite_divided_differences(
            &self.x_values,
            &self.y_values,
            &self.dy_values,
        );

        self.z_values = z;
        self.coefficients = coef;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn get_coefficients(&self) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y, dy) first.".to_string());
        }
        Ok(self.coefficients.clone())
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y, dy) first.".to_string());
        }
        Ok(hermite_evaluate(&self.z_values, &self.coefficients, x))
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y, dy) first.".to_string());
        }
        if xs.len() != out.len() {
            return Err("input and output slices must have the same length".to_string());
        }

        let mut i = 0;
        let n = xs.len();
        let z = &self.z_values;
        let coeff = &self.coefficients;

        while i + 1 < n {
            out[i] = hermite_evaluate(z, coeff, xs[i]);
            out[i + 1] = hermite_evaluate(z, coeff, xs[i + 1]);
            i += 2;
        }
        if i < n {
            out[i] = hermite_evaluate(z, coeff, xs[i]);
        }
        Ok(())
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(xs.len());
        let mut i = 0;
        let z = &self.z_values;
        let coeff = &self.coefficients;

        while i + 1 < xs.len() {
            out.push(hermite_evaluate(z, coeff, xs[i]));
            out.push(hermite_evaluate(z, coeff, xs[i + 1]));
            i += 2;
        }
        if i < xs.len() {
            out.push(hermite_evaluate(z, coeff, xs[i]));
        }
        Ok(out)
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

    for i in 0..n {
        z.push(x_values[i]);
        z.push(x_values[i]);
        q[2 * i * stride] = y_values[i];
        q[(2 * i + 1) * stride] = y_values[i];
    }

    for i in 0..n {
        q[(2 * i + 1) * stride + 1] = dy_values[i];
        if i > 0 {
            let idx_cur = 2 * i * stride;
            let idx_prev = (2 * i - 1) * stride;
            let denominator = z[2 * i] - z[2 * i - 1];
            q[2 * i * stride + 1] = (q[idx_cur] - q[idx_prev]) / denominator;
        }
    }

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
    for i in (0..n - 1).rev() {
        result = result * (x - z[i]) + coefficients[i];
    }
    result
}
