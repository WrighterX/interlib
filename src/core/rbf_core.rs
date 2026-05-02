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
        let mut x_sorted = xs.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..x_sorted.len().saturating_sub(1) {
            if x_sorted[i] == x_sorted[i + 1] {
                return Err(CoreError::DistinctNodesRequired { what: "x values" }.into());
            }
        }

        let weights = compute_rbf_weights(xs, ys, self.kernel, self.epsilon)?;
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
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(xs
            .iter()
            .map(|&value| {
                rbf_evaluate(
                    &self.x_values,
                    &self.weights,
                    self.kernel,
                    self.epsilon,
                    value,
                )
            })
            .collect())
    }
}

fn compute_rbf_weights(
    x_values: &[f64],
    y_values: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
) -> Result<Vec<f64>, String> {
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
