pub(crate) trait InterpolationCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String>;

    #[inline]
    fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let n = xs.len();
        let mut out = vec![0.0; n];
        self.fill_many(xs, &mut out)?;
        Ok(out)
    }

    #[inline]
    fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if xs.len() != out.len() {
            return Err("input and output slices must have the same length".to_string());
        }
        let mut i = 0;
        while i + 1 < xs.len() {
            out[i] = self.evaluate_single(xs[i])?;
            out[i + 1] = self.evaluate_single(xs[i + 1])?;
            i += 2;
        }
        if i < xs.len() {
            out[i] = self.evaluate_single(xs[i])?;
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn is_non_decreasing(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[0] <= w[1])
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_linear_system_gaussian(
    mut a: Vec<Vec<f64>>,
    mut b: Vec<f64>,
) -> Result<Vec<f64>, String> {
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
            return Err(format!("Singular matrix near pivot {}", k));
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
