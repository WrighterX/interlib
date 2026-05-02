use crate::core_error::CoreError;

#[derive(Clone, Debug)]
pub(crate) struct LinearCore {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    slopes: Vec<f64>,
    fitted: bool,
}

impl LinearCore {
    pub(crate) fn new() -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
            slopes: Vec::new(),
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

        // Sorting upfront makes the later segment lookup a binary search
        // instead of a linear scan. That is the main reason this core stays
        // fast even when wrappers call it repeatedly.
        let mut is_sorted = true;
        for i in 0..x.len() - 1 {
            if x[i] >= x[i + 1] {
                is_sorted = false;
                break;
            }
        }

        let (x, y) = if !is_sorted {
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

            let x_sorted = indices.iter().map(|&i| x[i]).collect();
            let y_sorted = indices.iter().map(|&i| y[i]).collect();
            (x_sorted, y_sorted)
        } else {
            (x, y)
        };

        for i in 0..x.len().saturating_sub(1) {
            if x[i] == x[i + 1] {
                return Err(CoreError::DistinctNodesRequired { what: "x values" }.into());
            }
        }

        self.slopes = (0..x.len() - 1)
            .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            .collect();
        self.x_values = x;
        self.y_values = y;
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(x, y) first.",
            }
            .into());
        }
        Ok(linear_interpolate_single(
            &self.x_values,
            &self.y_values,
            &self.slopes,
            x,
        ))
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0; xs.len()];
        self.fill_many(xs, &mut out)?;
        Ok(out)
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
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

        // For monotonic query grids, advance a single segment cursor instead
        // of running a binary search for every point.
        if is_non_decreasing(xs) {
            linear_fill_many_sorted(
                &self.x_values,
                &self.y_values,
                &self.slopes,
                xs,
                out,
            );
            return Ok(());
        }

        // Fallback path for unordered input.
        let mut i = 0;
        while i + 1 < xs.len() {
            out[i] = linear_interpolate_single(&self.x_values, &self.y_values, &self.slopes, xs[i]);
            out[i + 1] =
                linear_interpolate_single(&self.x_values, &self.y_values, &self.slopes, xs[i + 1]);
            i += 2;
        }
        if i < xs.len() {
            out[i] = linear_interpolate_single(&self.x_values, &self.y_values, &self.slopes, xs[i]);
        }
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
            return Err(format!(
                "y must have length {} (same as x), got {}",
                self.x_values.len(),
                y.len()
            ));
        }
        // Only the slopes change here. The x-grid remains valid, so there is
        // no need to re-run the entire fit logic.
        self.slopes = (0..self.x_values.len() - 1)
            .map(|i| (y[i + 1] - y[i]) / (self.x_values[i + 1] - self.x_values[i]))
            .collect();
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

        let idx = match self
            .x_values
            .binary_search_by(|v| v.partial_cmp(&x_new).unwrap())
        {
            Ok(_) => {
                return Err(format!("x_new = {} already exists in the data", x_new));
            }
            Err(i) => i,
        };

        self.x_values.insert(idx, x_new);
        self.y_values.insert(idx, y_new);

        let n = self.x_values.len();
        if n < 2 {
            self.slopes.clear();
            return Ok(());
        }

        // Only the touched neighborhood needs updating.
        if idx == 0 {
            let new_slope =
                (self.y_values[1] - self.y_values[0]) / (self.x_values[1] - self.x_values[0]);
            self.slopes.insert(0, new_slope);
        } else if idx == n - 1 {
            let new_slope = (self.y_values[idx] - self.y_values[idx - 1])
                / (self.x_values[idx] - self.x_values[idx - 1]);
            self.slopes.push(new_slope);
        } else {
            let slope_left = (self.y_values[idx] - self.y_values[idx - 1])
                / (self.x_values[idx] - self.x_values[idx - 1]);
            let slope_right = (self.y_values[idx + 1] - self.y_values[idx])
                / (self.x_values[idx + 1] - self.x_values[idx]);
            self.slopes[idx - 1] = slope_left;
            self.slopes.insert(idx, slope_right);
        }

        Ok(())
    }

    pub(crate) fn repr(&self) -> String {
        if self.fitted {
            format!(
                "LinearInterpolator(fitted with {} points, x range: [{:.2}, {:.2}])",
                self.x_values.len(),
                self.x_values.first().unwrap_or(&0.0),
                self.x_values.last().unwrap_or(&0.0)
            )
        } else {
            "LinearInterpolator(not fitted)".to_string()
        }
    }
}

#[inline]
fn linear_interpolate_single(x_values: &[f64], y_values: &[f64], slopes: &[f64], x: f64) -> f64 {
    let n = x_values.len();

    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return y_values[0];
    }
    if x <= x_values[0] {
        return y_values[0];
    }
    if x >= x_values[n - 1] {
        return y_values[n - 1];
    }

    // Binary search is enough because `fit()` keeps the x-grid sorted.
    let idx = match x_values.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(i) => {
            if i >= n - 1 {
                i - 1
            } else {
                i
            }
        }
        Err(i) => {
            if i > 0 {
                i - 1
            } else {
                0
            }
        }
    };

    // One slope and one left endpoint fully determine the segment value.
    y_values[idx] + slopes[idx] * (x - x_values[idx])
}

#[inline]
fn is_non_decreasing(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[0] <= w[1])
}

fn linear_fill_many_sorted(
    x_values: &[f64],
    y_values: &[f64],
    slopes: &[f64],
    xs: &[f64],
    out: &mut [f64],
) {
    let n = x_values.len();
    if n == 0 {
        out.fill(f64::NAN);
        return;
    }
    if n == 1 {
        out.fill(y_values[0]);
        return;
    }

    let mut seg = 0usize;
    for (i, &x) in xs.iter().enumerate() {
        if x <= x_values[0] {
            out[i] = y_values[0];
            continue;
        }
        if x >= x_values[n - 1] {
            out[i] = y_values[n - 1];
            continue;
        }

        while seg + 1 < n - 1 && x > x_values[seg + 1] {
            seg += 1;
        }
        out[i] = y_values[seg] + slopes[seg] * (x - x_values[seg]);
    }
}

#[cfg(test)]
mod tests {
    use super::LinearCore;

    #[test]
    fn fit_sorts_input_and_evaluates() {
        let mut core = LinearCore::new();
        core.fit(vec![2.0, 0.0, 1.0], vec![4.0, 0.0, 1.0]).unwrap();
        assert_eq!(core.evaluate_single(0.5).unwrap(), 0.5);
        assert_eq!(core.evaluate_single(-1.0).unwrap(), 0.0);
        assert_eq!(core.evaluate_single(3.0).unwrap(), 4.0);
    }

    #[test]
    fn update_y_and_add_point_work() {
        let mut core = LinearCore::new();
        core.fit(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();
        core.update_y(vec![0.0, 2.0, 6.0]).unwrap();
        assert_eq!(core.evaluate_single(0.5).unwrap(), 1.0);

        core.add_point(1.5, 3.0).unwrap();
        assert_eq!(core.evaluate_single(1.5).unwrap(), 3.0);
        assert!(core.repr().contains("fitted with 4 points"));
    }

    #[test]
    fn evaluate_many_and_fill_many_match_scalar_path() {
        let mut core = LinearCore::new();
        core.fit(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 4.0, 9.0])
            .unwrap();

        let xs = vec![-1.0, 0.5, 1.5, 3.0, 4.0];
        let scalar: Vec<f64> = xs
            .iter()
            .map(|&x| core.evaluate_single(x).unwrap())
            .collect();

        let many = core.evaluate_many(&xs).unwrap();
        assert_eq!(many, scalar);

        let mut out = vec![0.0; xs.len()];
        core.fill_many(&xs, &mut out).unwrap();
        assert_eq!(out, scalar);

        let xs_unsorted = vec![1.5, -1.0, 4.0, 0.5, 3.0];
        let scalar_unsorted: Vec<f64> = xs_unsorted
            .iter()
            .map(|&x| core.evaluate_single(x).unwrap())
            .collect();
        let mut out_unsorted = vec![0.0; xs_unsorted.len()];
        core.fill_many(&xs_unsorted, &mut out_unsorted).unwrap();
        assert_eq!(out_unsorted, scalar_unsorted);
    }
}
