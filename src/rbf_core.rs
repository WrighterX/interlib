/// The supported radial basis function kernels.
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
            return Err("epsilon must be positive".into());
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
            return Err("x and y must have the same length".into());
        }
        if xs.is_empty() {
            return Err("x and y cannot be empty".into());
        }

        let weights = compute_rbf_weights(xs, ys, self.kernel, self.epsilon)?;
        self.x_values = xs.to_vec();
        self.weights = weights;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn evaluate_single(&self, value: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err("Interpolator not fitted".into());
        }
        Ok(rbf_evaluate(
            &self.x_values,
            &self.weights,
            self.kernel,
            self.epsilon,
            value,
        ))
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("Interpolator not fitted".into());
        }
        Ok(xs
            .iter()
            .map(|&value| {
                rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, value)
            })
            .collect())
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if xs.len() != out.len() {
            return Err("Input/output length mismatch".into());
        }
        if !self.fitted {
            return Err("Interpolator not fitted".into());
        }
        for (value, slot) in xs.iter().zip(out.iter_mut()) {
            *slot = rbf_evaluate(&self.x_values, &self.weights, self.kernel, self.epsilon, *value);
        }
        Ok(())
    }

    pub(crate) fn weights(&self) -> Result<&[f64], String> {
        if !self.fitted {
            return Err("Interpolator not fitted".into());
        }
        Ok(&self.weights)
    }

    pub(crate) fn kernel(&self) -> RBFKernel {
        self.kernel
    }

    pub(crate) fn epsilon(&self) -> f64 {
        self.epsilon
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
            return Err("Interpolator not fitted".into());
        }
        if out.len() != self.weights.len() {
            return Err("Output length mismatch".into());
        }
        out.copy_from_slice(&self.weights);
        Ok(())
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

    solve_linear_system(matrix, y_values.to_vec())
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();

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
            return Err("Matrix is singular or nearly singular".into());
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

fn rbf_evaluate(
    x_values: &[f64],
    weights: &[f64],
    kernel: RBFKernel,
    epsilon: f64,
    x: f64,
) -> f64 {
    let mut result = 0.0;
    for idx in 0..x_values.len() {
        let r = (x - x_values[idx]).abs();
        result += weights[idx] * kernel.evaluate(r, epsilon);
    }
    result
}
