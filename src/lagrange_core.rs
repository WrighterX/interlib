/// Lagrange interpolation core that everything (Python/FFI) shares.
pub(crate) struct LagrangeCore {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    weights: Vec<f64>,
    fitted: bool,
}

impl LagrangeCore {
    pub(crate) fn new() -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
            weights: Vec::new(),
            fitted: false,
        }
    }

    pub(crate) fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("x and y must have the same length".into());
        }
        if x.is_empty() {
            return Err("x and y cannot be empty".into());
        }

        self.weights = compute_barycentric_weights(&x);
        self.x_values = x;
        self.y_values = y;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn update_y(&mut self, y: Vec<f64>) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".into());
        }
        if y.len() != self.x_values.len() {
            return Err("New y must have the same length as x".into());
        }
        self.y_values = y;
        Ok(())
    }

    pub(crate) fn add_point(&mut self, x_new: f64, y_new: f64) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".into());
        }

        for x in &self.x_values {
            if (*x - x_new).abs() < f64::EPSILON {
                return Err(format!("x_new {} already exists", x_new));
            }
        }

        for j in 0..self.weights.len() {
            self.weights[j] /= self.x_values[j] - x_new;
        }
        let mut w_new = 1.0;
        for &x in &self.x_values {
            w_new *= x_new - x;
        }
        w_new = 1.0 / w_new;

        self.x_values.push(x_new);
        self.y_values.push(y_new);
        self.weights.push(w_new);
        Ok(())
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".into());
        }
        Ok(barycentric_eval(
            &self.x_values,
            &self.y_values,
            &self.weights,
            x,
        ))
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".into());
        }
        if xs.len() != out.len() {
            return Err("input and output slices must match".into());
        }

        let n = xs.len();
        let mut i = 0;
        while i + 1 < n {
            out[i] = barycentric_eval(&self.x_values, &self.y_values, &self.weights, xs[i]);
            out[i + 1] = barycentric_eval(&self.x_values, &self.y_values, &self.weights, xs[i + 1]);
            i += 2;
        }
        if i < n {
            out[i] = barycentric_eval(&self.x_values, &self.y_values, &self.weights, xs[i]);
        }
        Ok(())
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = Vec::with_capacity(xs.len());
        let mut i = 0;
        while i + 1 < xs.len() {
            out.push(barycentric_eval(
                &self.x_values,
                &self.y_values,
                &self.weights,
                xs[i],
            ));
            out.push(barycentric_eval(
                &self.x_values,
                &self.y_values,
                &self.weights,
                xs[i + 1],
            ));
            i += 2;
        }
        if i < xs.len() {
            out.push(barycentric_eval(
                &self.x_values,
                &self.y_values,
                &self.weights,
                xs[i],
            ));
        }
        Ok(out)
    }

    pub(crate) fn repr(&self) -> String {
        if self.fitted {
            format!(
                "LagrangeInterpolator(barycentric, fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "LagrangeInterpolator(barycentric, not fitted)".to_string()
        }
    }
}

fn compute_barycentric_weights(x_values: &[f64]) -> Vec<f64> {
    let n = x_values.len();
    let mut weights = vec![1.0; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                weights[j] *= x_values[j] - x_values[k];
            }
        }
        weights[j] = 1.0 / weights[j];
    }
    weights
}

fn barycentric_eval(x_values: &[f64], y_values: &[f64], weights: &[f64], x: f64) -> f64 {
    let mut numer = 0.0;
    let mut denom = 0.0;
    for j in 0..x_values.len() {
        let diff = x - x_values[j];
        if diff == 0.0 {
            return y_values[j];
        }
        let term = weights[j] / diff;
        numer += term * y_values[j];
        denom += term;
    }
    numer / denom
}
