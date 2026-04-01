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
            return Err("x and y must have the same length".to_string());
        }
        if x.is_empty() {
            return Err("x and y cannot be empty".to_string());
        }

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
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        Ok(linear_interpolate_single(
            &self.x_values,
            &self.y_values,
            &self.slopes,
            x,
        ))
    }

    pub(crate) fn evaluate_many(&self, xs: &[f64]) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }

        let n = xs.len();
        let mut results = Vec::with_capacity(n);
        let mut i = 0;
        while i + 1 < n {
            results.push(linear_interpolate_single(
                &self.x_values,
                &self.y_values,
                &self.slopes,
                xs[i],
            ));
            results.push(linear_interpolate_single(
                &self.x_values,
                &self.y_values,
                &self.slopes,
                xs[i + 1],
            ));
            i += 2;
        }
        if i < n {
            results.push(linear_interpolate_single(
                &self.x_values,
                &self.y_values,
                &self.slopes,
                xs[i],
            ));
        }
        Ok(results)
    }

    pub(crate) fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        if xs.len() != out.len() {
            return Err("input and output slices must have the same length".to_string());
        }

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
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
        }
        if y.len() != self.x_values.len() {
            return Err(format!(
                "y must have length {} (same as x), got {}",
                self.x_values.len(),
                y.len()
            ));
        }
        self.slopes = (0..self.x_values.len() - 1)
            .map(|i| (y[i + 1] - y[i]) / (self.x_values[i + 1] - self.x_values[i]))
            .collect();
        self.y_values = y;
        Ok(())
    }

    pub(crate) fn add_point(&mut self, x_new: f64, y_new: f64) -> Result<(), String> {
        if !self.fitted {
            return Err("Interpolator not fitted. Call fit(x, y) first.".to_string());
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

    y_values[idx] + slopes[idx] * (x - x_values[idx])
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
}
