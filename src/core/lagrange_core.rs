use crate::core::core_error::CoreError;
use crate::core::core_trait::InterpolationCore;

pub(crate) struct LagrangeCore {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    weights: Vec<f64>,
    weighted_y: Vec<f64>,
    fitted: bool,
}

impl LagrangeCore {
    pub(crate) fn new() -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
            weights: Vec::new(),
            weighted_y: Vec::new(),
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

        self.weights = compute_barycentric_weights(&x);
        self.weighted_y = compute_weighted_y(&self.weights, &y);
        self.x_values = x;
        self.y_values = y;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn update_y(&mut self, y: Vec<f64>) -> Result<(), String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        if y.len() != self.x_values.len() {
            return Err("New y must have the same length as x".into());
        }
        self.weighted_y = compute_weighted_y(&self.weights, &y);
        self.y_values = y;
        Ok(())
    }

    pub(crate) fn add_point(&mut self, x_new: f64, y_new: f64) -> Result<(), String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
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
        self.weighted_y = compute_weighted_y(&self.weights, &self.y_values);
        Ok(())
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

impl InterpolationCore for LagrangeCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(barycentric_eval(
            &self.x_values,
            &self.y_values,
            &self.weights,
            &self.weighted_y,
            x,
        ))
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
            let values = barycentric_eval2(
                &self.x_values,
                &self.y_values,
                &self.weights,
                &self.weighted_y,
                xs[i],
                xs[i + 1],
            );
            out[i] = values[0];
            out[i + 1] = values[1];
            i += 2;
        }
        if i < n {
            out[i] = barycentric_eval(
                &self.x_values,
                &self.y_values,
                &self.weights,
                &self.weighted_y,
                xs[i],
            );
        }
        Ok(())
    }
}

fn compute_barycentric_weights(x_values: &[f64]) -> Vec<f64> {
    let n = x_values.len();
    let mut weights = vec![1.0; n];
    // O(n^2) precompute, but evaluation becomes O(n) with no polynomial build.
    // Each node pair contributes to two weights with opposite signs.
    for j in 0..n {
        for k in j + 1..n {
            let diff = x_values[j] - x_values[k];
            weights[j] *= diff;
            weights[k] *= -diff;
        }
    }
    for weight in &mut weights {
        *weight = 1.0 / *weight;
    }
    weights
}

fn compute_weighted_y(weights: &[f64], y_values: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(y_values.iter())
        .map(|(&weight, &y)| weight * y)
        .collect()
}

fn barycentric_eval(
    x_values: &[f64],
    y_values: &[f64],
    weights: &[f64],
    weighted_y: &[f64],
    x: f64,
) -> f64 {
    let mut numer = 0.0;
    let mut denom = 0.0;
    for j in 0..x_values.len() {
        let diff = x - x_values[j];
        if diff == 0.0 {
            return y_values[j];
        }
        // The barycentric ratio keeps the interpolation numerically stable.
        numer += weighted_y[j] / diff;
        denom += weights[j] / diff;
    }
    numer / denom
}

fn barycentric_eval2(
    x_values: &[f64],
    y_values: &[f64],
    weights: &[f64],
    weighted_y: &[f64],
    x0: f64,
    x1: f64,
) -> [f64; 2] {
    let mut numer0 = 0.0;
    let mut denom0 = 0.0;
    let mut numer1 = 0.0;
    let mut denom1 = 0.0;
    let mut exact0 = 0.0;
    let mut exact1 = 0.0;
    let mut has_exact0 = false;
    let mut has_exact1 = false;

    for j in 0..x_values.len() {
        let node = x_values[j];
        let y = y_values[j];
        let weight = weights[j];
        let weighted = weighted_y[j];

        let diff0 = x0 - node;
        if diff0 == 0.0 {
            if !has_exact0 {
                exact0 = y;
                has_exact0 = true;
            }
        } else {
            numer0 += weighted / diff0;
            denom0 += weight / diff0;
        }

        let diff1 = x1 - node;
        if diff1 == 0.0 {
            if !has_exact1 {
                exact1 = y;
                has_exact1 = true;
            }
        } else {
            numer1 += weighted / diff1;
            denom1 += weight / diff1;
        }
    }

    [
        if has_exact0 { exact0 } else { numer0 / denom0 },
        if has_exact1 { exact1 } else { numer1 / denom1 },
    ]
}
