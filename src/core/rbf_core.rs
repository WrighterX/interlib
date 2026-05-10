use crate::core::core_error::CoreError;
use crate::core::core_trait::{InterpolationCore, solve_linear_system_gaussian};

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RBFKernel {
    Gaussian = 0,
    Multiquadric = 1,
    InverseMultiquadric = 2,
    ThinPlateSpline = 3,
    Linear = 4,
}

impl RBFKernel {
    fn evaluate(&self, r: f64, epsilon: f64) -> f64 {
        match self {
            RBFKernel::Gaussian => (-epsilon * epsilon * r * r).exp(),
            RBFKernel::Multiquadric => (1.0 + (epsilon * r).powi(2)).sqrt(),
            RBFKernel::InverseMultiquadric => 1.0 / (1.0 + (epsilon * r).powi(2)).sqrt(),
            RBFKernel::ThinPlateSpline => {
                if r == 0.0 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
            RBFKernel::Linear => r,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            RBFKernel::Gaussian => "gaussian",
            RBFKernel::Multiquadric => "multiquadric",
            RBFKernel::InverseMultiquadric => "inverse_multiquadric",
            RBFKernel::ThinPlateSpline => "thin_plate_spline",
            RBFKernel::Linear => "linear",
        }
    }

    pub(crate) fn from_str(name: &str) -> Result<Self, String> {
        let normalized = name.trim().to_lowercase();
        match normalized.as_str() {
            "gaussian" => Ok(RBFKernel::Gaussian),
            "multiquadric" => Ok(RBFKernel::Multiquadric),
            "inverse_multiquadric" | "inverse multiquadric" => Ok(RBFKernel::InverseMultiquadric),
            "thin_plate_spline" | "thin plate spline" => Ok(RBFKernel::ThinPlateSpline),
            "linear" => Ok(RBFKernel::Linear),
            other => Err(format!(
                "Unknown kernel type '{}'. Available: gaussian, multiquadric, inverse_multiquadric, thin_plate_spline, linear",
                other
            )),
        }
    }

    pub(crate) fn from_id(value: i32) -> Result<Self, &'static str> {
        match value {
            0 => Ok(RBFKernel::Gaussian),
            1 => Ok(RBFKernel::Multiquadric),
            2 => Ok(RBFKernel::InverseMultiquadric),
            3 => Ok(RBFKernel::ThinPlateSpline),
            4 => Ok(RBFKernel::Linear),
            _ => Err("Unknown kernel id"),
        }
    }
}

pub(crate) struct RBFCore {
    x_values: Vec<f64>,
    weights: Vec<f64>,
    uniform_step: Option<f64>,
    kernel: RBFKernel,
    epsilon: f64,
    fitted: bool,
}

impl RBFCore {
    pub(crate) fn new(kernel: RBFKernel, epsilon: f64) -> Result<Self, String> {
        if epsilon <= 0.0 {
            return Err(CoreError::Message("epsilon must be positive".to_string()).into());
        }

        Ok(Self {
            x_values: Vec::new(),
            weights: Vec::new(),
            uniform_step: None,
            kernel,
            epsilon,
            fitted: false,
        })
    }

    pub(crate) fn fit(&mut self, xs: &[f64], ys: &[f64]) -> Result<(), String> {
        if xs.len() != ys.len() {
            return Err(CoreError::LengthMismatch {
                left_name: "x",
                left: xs.len(),
                right_name: "y",
                right: ys.len(),
            }
            .into());
        }
        if xs.is_empty() {
            return Err(CoreError::EmptyInput { what: "x and y" }.into());
        }
        if crate::core::core_trait::is_non_decreasing(xs) {
            for i in 0..xs.len().saturating_sub(1) {
                if xs[i] == xs[i + 1] {
                    return Err(CoreError::DistinctNodesRequired { what: "x values" }.into());
                }
            }
        } else {
            let mut x_sorted = xs.to_vec();
            x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 0..x_sorted.len().saturating_sub(1) {
                if x_sorted[i] == x_sorted[i + 1] {
                    return Err(CoreError::DistinctNodesRequired { what: "x values" }.into());
                }
            }
        }

        let weights = compute_rbf_weights(xs, ys, self.kernel, self.epsilon)?;
        self.uniform_step = detect_uniform_step(xs);
        self.x_values = xs.to_vec();
        self.weights = weights;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn weights(&self) -> Result<&[f64], String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(&self.weights)
    }

    pub(crate) fn repr(&self) -> String {
        if self.fitted {
            format!(
                "RBFInterpolator(kernel='{}', epsilon={:.2}, fitted with {} points)",
                self.kernel.name(),
                self.epsilon,
                self.x_values.len()
            )
        } else {
            format!(
                "RBFInterpolator(kernel='{}', epsilon={:.2}, not fitted)",
                self.kernel.name(),
                self.epsilon
            )
        }
    }

    pub(crate) fn point_count(&self) -> usize {
        self.x_values.len()
    }

    pub(crate) fn fill_weights(&self, out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        if out.len() != self.weights.len() {
            return Err(CoreError::LengthMismatch {
                left_name: "weights",
                left: self.weights.len(),
                right_name: "output",
                right: out.len(),
            }
            .into());
        }
        out.copy_from_slice(&self.weights);
        Ok(())
    }
}

impl InterpolationCore for RBFCore {
    fn evaluate_single(&self, value: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(rbf_evaluate(
            &self.x_values,
            &self.weights,
            self.kernel,
            self.epsilon,
            value,
        ))
    }

    fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if xs.len() != out.len() {
            return Err(CoreError::LengthMismatch {
                left_name: "input",
                left: xs.len(),
                right_name: "output",
                right: out.len(),
            }
            .into());
        }
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        if self.kernel == RBFKernel::Gaussian {
            if let Some(step) = self.uniform_step {
                rbf_fill_many_gaussian_uniform_centers(
                    &self.x_values,
                    &self.weights,
                    self.epsilon,
                    step,
                    xs,
                    out,
                );
            } else {
                rbf_fill_many_gaussian(&self.x_values, &self.weights, self.epsilon, xs, out);
            }
            return Ok(());
        }

        for (value, slot) in xs.iter().zip(out.iter_mut()) {
            *slot = rbf_evaluate(
                &self.x_values,
                &self.weights,
                self.kernel,
                self.epsilon,
                *value,
            );
        }
        Ok(())
    }

    fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0; xs.len()];
        self.fill_many(xs, &mut out)?;
        Ok(out)
    }
}

fn detect_uniform_step(x_values: &[f64]) -> Option<f64> {
    if x_values.len() < 3 {
        return None;
    }
    let step = x_values[1] - x_values[0];
    if step <= 0.0 || !step.is_finite() {
        return None;
    }
    let tolerance = step.abs() * 1e-10 + f64::EPSILON;
    for pair in x_values.windows(2).skip(1) {
        if ((pair[1] - pair[0]) - step).abs() > tolerance {
            return None;
        }
    }
    Some(step)
}

fn compute_rbf_weights(
    x_values: &[f64],
    y_values: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> Result<Vec<f64>, String> {
    if kernel == RBFKernel::Gaussian {
        return compute_gaussian_rbf_weights_flat(x_values, y_values, epsilon);
    }

    let n = x_values.len();
    let mut matrix = vec![vec![0.0; n]; n];
    let diag = kernel.evaluate(0.0, epsilon);
    for i in 0..n {
        matrix[i][i] = diag;
        for j in i + 1..n {
            let r = (x_values[i] - x_values[j]).abs();
            let val = kernel.evaluate(r, epsilon);
            matrix[i][j] = val;
            matrix[j][i] = val;
        }
    }

    let result = solve_linear_system_gaussian(matrix, y_values.to_vec())?;
    Ok(result)
}

fn compute_gaussian_rbf_weights_flat(
    x_values: &[f64],
    y_values: &[f64],
    epsilon: f64,
) -> Result<Vec<f64>, String> {
    let n = x_values.len();
    let mut matrix = vec![0.0; n * n];
    let epsilon_sq = epsilon * epsilon;
    for i in 0..n {
        matrix[i * n + i] = 1.0;
        for j in i + 1..n {
            let diff = x_values[i] - x_values[j];
            let val = (-(epsilon_sq * diff * diff)).exp();
            matrix[i * n + j] = val;
            matrix[j * n + i] = val;
        }
    }

    solve_linear_system_gaussian_flat(matrix, y_values.to_vec(), n)
}

fn solve_linear_system_gaussian_flat(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    n: usize,
) -> Result<Vec<f64>, String> {
    for k in 0..n {
        let mut max_idx = k;
        let mut max_val = a[k * n + k].abs();
        for i in k + 1..n {
            let value = a[i * n + k].abs();
            if value > max_val {
                max_val = value;
                max_idx = i;
            }
        }
        if max_idx != k {
            for j in k..n {
                a.swap(k * n + j, max_idx * n + j);
            }
            b.swap(k, max_idx);
        }
        let pivot = a[k * n + k];
        if pivot.abs() < 1e-12 {
            return Err(format!("Singular matrix near pivot {}", k));
        }
        for i in k + 1..n {
            let row_base = i * n;
            let factor = a[row_base + k] / pivot;
            a[row_base + k] = 0.0;
            for j in k + 1..n {
                a[row_base + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut value = b[i];
        for j in i + 1..n {
            value -= a[i * n + j] * x[j];
        }
        x[i] = value / a[i * n + i];
    }
    Ok(x)
}

fn rbf_fill_many_gaussian_uniform_centers(
    x_values: &[f64],
    weights: &[f64],
    epsilon: f64,
    step: f64,
    xs: &[f64],
    out: &mut [f64],
) {
    let epsilon_sq = epsilon * epsilon;
    let first = x_values[0];
    let last_idx = x_values.len() - 1;
    let inv_step = 1.0 / step;
    let step_sq = epsilon_sq * step * step;
    let two_epsilon_sq_step = 2.0 * epsilon_sq * step;
    let step_kernel_decay = (-2.0 * step_sq).exp();

    for (slot, &x) in out.iter_mut().zip(xs.iter()) {
        let nearest = ((x - first) * inv_step).round();
        let center_idx = if nearest <= 0.0 {
            0
        } else if nearest >= last_idx as f64 {
            last_idx
        } else {
            nearest as usize
        };

        let center = x_values[center_idx];
        let d = x - center;
        let kernel_center = (-(epsilon_sq * d * d)).exp();
        let mut result = weights[center_idx] * kernel_center;

        let mut kernel = kernel_center;
        let mut ratio = (two_epsilon_sq_step * d - step_sq).exp();
        for j in center_idx + 1..x_values.len() {
            kernel *= ratio;
            result += weights[j] * kernel;
            ratio *= step_kernel_decay;
        }

        let mut kernel = kernel_center;
        let mut ratio = (-two_epsilon_sq_step * d - step_sq).exp();
        for j in (0..center_idx).rev() {
            kernel *= ratio;
            result += weights[j] * kernel;
            ratio *= step_kernel_decay;
        }

        *slot = result;
    }
}

fn rbf_fill_many_gaussian(
    x_values: &[f64],
    weights: &[f64],
    epsilon: f64,
    xs: &[f64],
    out: &mut [f64],
) {
    let epsilon_sq = epsilon * epsilon;
    let mut i = 0;
    while i + 3 < xs.len() {
        let x0 = xs[i];
        let x1 = xs[i + 1];
        let x2 = xs[i + 2];
        let x3 = xs[i + 3];
        let mut r0 = 0.0;
        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut r3 = 0.0;

        for idx in 0..x_values.len() {
            let center = x_values[idx];
            let weight = weights[idx];

            let dx0 = x0 - center;
            r0 += weight * (-(epsilon_sq * dx0 * dx0)).exp();

            let dx1 = x1 - center;
            r1 += weight * (-(epsilon_sq * dx1 * dx1)).exp();

            let dx2 = x2 - center;
            r2 += weight * (-(epsilon_sq * dx2 * dx2)).exp();

            let dx3 = x3 - center;
            r3 += weight * (-(epsilon_sq * dx3 * dx3)).exp();
        }

        out[i] = r0;
        out[i + 1] = r1;
        out[i + 2] = r2;
        out[i + 3] = r3;
        i += 4;
    }

    while i < xs.len() {
        out[i] = rbf_evaluate(x_values, weights, RBFKernel::Gaussian, epsilon, xs[i]);
        i += 1;
    }
}

fn rbf_evaluate(x_values: &[f64], weights: &[f64], kernel: RBFKernel, epsilon: f64, x: f64) -> f64 {
    let mut result = 0.0;
    for idx in 0..x_values.len() {
        let r = (x - x_values[idx]).abs();
        result += weights[idx] * kernel.evaluate(r, epsilon);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{RBFCore, RBFKernel};
    use crate::core::core_trait::InterpolationCore;
    use crate::core::core_trait::solve_linear_system_gaussian;

    #[test]
    fn fit_and_evaluate_small_case() {
        let mut core = RBFCore::new(RBFKernel::Gaussian, 1.0).unwrap();
        core.fit(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap();

        let v = core.evaluate_single(1.0).unwrap();
        assert!((v - 1.0).abs() < 1e-7);

        let many = core.evaluate_many(&[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(many.len(), 3);
    }

    #[test]
    fn duplicate_x_is_rejected() {
        let mut core = RBFCore::new(RBFKernel::Gaussian, 1.0).unwrap();
        let err = core.fit(&[0.0, 1.0, 1.0], &[0.0, 1.0, 4.0]).unwrap_err();
        assert!(err.contains("distinct"));
    }

    #[test]
    fn singular_system_reports_diagnostic_error() {
        let err =
            solve_linear_system_gaussian(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 2.0])
                .unwrap_err();
        assert!(err.contains("Singular matrix"));
    }
}
