/// Lagrange Interpolation Module (Barycentric Form)
///
/// This module implements Lagrange polynomial interpolation using the
/// **barycentric formula**, a numerically stable and computationally
/// efficient reformulation of the classical Lagrange form.
///
/// # Mathematical Background
///
/// Given n+1 points (x₀, y₀), ..., (xₙ, yₙ), the classical Lagrange
/// polynomial P(x) = Σ yᵢ · Lᵢ(x) can be rewritten using barycentric
/// weights wⱼ = 1 / Π(k≠j)(xⱼ - xₖ) as:
///
///            Σ  wⱼ/(x - xⱼ) · yⱼ
/// P(x) =   ─────────────────────────
///            Σ  wⱼ/(x - xⱼ)
///
/// This is the "second (true) form of the barycentric formula"
/// (Rutishauser, as cited in Berrut & Trefethen, 2004).
///
/// # Complexity
///
/// - **Weight precomputation**: O(n²) — performed once at fit time.
///   Crucially, the weights depend only on the nodes xⱼ, *not* on the
///   data yⱼ. This means the same fitted interpolator can evaluate any
///   number of different functions defined on those nodes in O(n) each.
/// - **Evaluation**: O(n) per point — a significant improvement over the
///   O(n²) cost of direct Lagrange basis evaluation.
/// - **Update**: Adding a single new node costs O(n), not a full
///   recomputation (see the `add_point` method).
///
/// # Numerical Stability
///
/// The barycentric formula is unconditionally stable in its first form
/// and stable in its second form (the one used here) provided the nodes
/// are well-distributed (e.g., Chebyshev points). When x coincides
/// exactly with a node xₖ, the formula would divide by zero; this is
/// handled explicitly by returning yₖ directly.
///
/// # Use Cases
///
/// - Interpolation at many points after a single fit (amortised O(n))
/// - Re-evaluation with different y-data on the same nodes (O(n) each)
/// - Incremental addition of nodes without full recomputation
/// - Numerically stable polynomial interpolation for moderate-to-large n
///
/// # Examples
///
/// ```python
/// from interlib import LagrangeInterpolator
///
/// interp = LagrangeInterpolator()
///
/// x = [0.0, 1.0, 2.0, 3.0]
/// y = [0.0, 1.0, 4.0, 9.0]
/// interp.fit(x, y)
///
/// result  = interp(1.5)          # single point
/// results = interp([1.5, 2.5])   # multiple points
/// ```
///
/// # References
///
/// Berrut, J.-P. and Trefethen, L. N., "Barycentric Lagrange
/// Interpolation," SIAM Review, Vol. 46, No. 3, pp. 501–517, 2004.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Precompute the barycentric weights for a given set of nodes.
///
/// Implements equation (3.2) from Berrut & Trefethen:
///     wⱼ = 1 / Π(k≠j)(xⱼ - xₖ)
///
/// The weights are independent of any function data and need only be
/// computed once per set of nodes.
///
/// # Arguments
///
/// * `x_values` - The interpolation nodes xⱼ
///
/// # Returns
///
/// A vector of barycentric weights wⱼ, one per node.
fn compute_barycentric_weights(x_values: &[f64]) -> Vec<f64> {
    let n = x_values.len();
    let mut weights = vec![1.0; n];

    for j in 0..n {
        for k in 0..n {
            if k != j {
                weights[j] *= x_values[j] - x_values[k];
            }
        }
        // wⱼ = 1 / Π(k≠j)(xⱼ - xₖ)
        weights[j] = 1.0 / weights[j];
    }

    weights
}

/// Evaluate the interpolant at a single point using the second
/// barycentric formula (equation 4.2 in Berrut & Trefethen):
///
///            Σ  wⱼ/(x - xⱼ) · yⱼ
/// P(x) =   ─────────────────────────
///            Σ  wⱼ/(x - xⱼ)
///
/// If x coincides exactly with a node xₖ, the corresponding yₖ is
/// returned directly, avoiding division by zero. This is the fix
/// described in Section 7 of the paper.
///
/// # Arguments
///
/// * `x_values`  - The interpolation nodes
/// * `y_values`  - The data values at each node
/// * `weights`   - Precomputed barycentric weights
/// * `x`         - The point at which to evaluate
///
/// # Returns
///
/// The interpolated value P(x).
fn barycentric_eval(
    x_values: &[f64],
    y_values: &[f64],
    weights: &[f64],
    x: f64,
) -> f64 {
    let mut numer = 0.0;
    let mut denom = 0.0;

    for j in 0..x_values.len() {
        let diff = x - x_values[j];

        // Exact node hit — return the known value directly.
        // This handles the division-by-zero case described in Section 7.
        if diff == 0.0 {
            return y_values[j];
        }

        let term = weights[j] / diff;
        numer += term * y_values[j];
        denom += term;
    }

    numer / denom
}

/// Lagrange Polynomial Interpolator (Barycentric Form)
///
/// A stateful interpolator that precomputes barycentric weights at fit
/// time, then evaluates the interpolant in O(n) per point.
///
/// # Attributes
///
/// * `x_values` - Stored interpolation nodes
/// * `y_values` - Stored data values
/// * `weights`  - Precomputed barycentric weights (depend only on nodes)
/// * `fitted`   - Whether the interpolator has been fitted with data
#[pyclass]
pub struct LagrangeInterpolator {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    weights: Vec<f64>,
    fitted: bool,
}

#[pymethods]
impl LagrangeInterpolator {
    /// Create a new Lagrange interpolator.
    ///
    /// Returns
    /// -------
    /// LagrangeInterpolator
    ///     A new, unfitted interpolator instance.
    #[new]
    pub fn new() -> Self {
        LagrangeInterpolator {
            x_values: Vec::new(),
            y_values: Vec::new(),
            weights: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the interpolator with data points.
    ///
    /// Stores the nodes and data, and precomputes the barycentric
    /// weights in O(n²). Because the weights depend only on the nodes,
    /// subsequent calls to `update_y` can change the data in O(n)
    /// without recomputing weights.
    ///
    /// Parameters
    /// ----------
    /// x : list of float
    ///     X coordinates (nodes) of the data points. Must be distinct.
    /// y : list of float
    ///     Y coordinates (data values) at each node.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If x and y have different lengths, or if either is empty.
    ///
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        if x.len() != y.len() {
            return Err(PyValueError::new_err(
                "x and y must have the same length",
            ));
        }
        if x.is_empty() {
            return Err(PyValueError::new_err(
                "x and y cannot be empty",
            ));
        }

        // Precompute weights — O(n²), but only depends on the nodes.
        self.weights = compute_barycentric_weights(&x);
        self.x_values = x;
        self.y_values = y;
        self.fitted = true;
        Ok(())
    }

    /// Replace the data values without recomputing weights.
    ///
    /// Because barycentric weights depend only on the nodes (not the
    /// data), swapping in new y-values is O(n). This is the key
    /// advantage noted in Section 3 of Berrut & Trefethen: "the
    /// quantities that have to be computed in O(n²) operations do not
    /// depend on the data fⱼ."
    ///
    /// Parameters
    /// ----------
    /// y : list of float
    ///     New data values. Must have the same length as the original x.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted, or if the length
    ///     of y does not match the number of nodes.
    ///
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp.update_y([1.0, 2.0, 5.0])   # O(n), no weight recomputation
    /// >>> interp(1.5)
    pub fn update_y(&mut self, y: Vec<f64>) -> PyResult<()> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first.",
            ));
        }
        if y.len() != self.x_values.len() {
            return Err(PyValueError::new_err(
                "New y must have the same length as x",
            ));
        }
        self.y_values = y;
        Ok(())
    }

    /// Add a new data point, updating weights incrementally in O(n).
    ///
    /// Implements the O(n) update procedure described in Section 3 of
    /// Berrut & Trefethen: each existing weight wⱼ is divided by
    /// (xⱼ - x_{n+1}), and the new weight w_{n+1} is computed from
    /// scratch — both steps cost O(n).
    ///
    /// Parameters
    /// ----------
    /// x_new : float
    ///     The new node. Must be distinct from all existing nodes.
    /// y_new : float
    ///     The data value at the new node.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted.
    ///
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp.add_point(3.0, 9.0)   # O(n) update
    /// >>> interp(2.5)
    pub fn add_point(&mut self, x_new: f64, y_new: f64) -> PyResult<()> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first.",
            ));
        }

        // Update existing weights: wⱼ ← wⱼ / (xⱼ - x_{new})
        for j in 0..self.x_values.len() {
            self.weights[j] /= self.x_values[j] - x_new;
        }

        // Compute the new weight: w_{new} = 1 / Π(k)(x_{new} - xₖ)
        let mut w_new = 1.0;
        for &xk in &self.x_values {
            w_new *= x_new - xk;
        }
        w_new = 1.0 / w_new;

        self.x_values.push(x_new);
        self.y_values.push(y_new);
        self.weights.push(w_new);
        Ok(())
    }

    /// Evaluate the interpolation at one or more points.
    ///
    /// Uses the second barycentric formula for O(n) evaluation per
    /// point. Exact node hits are handled without division by zero.
    ///
    /// Parameters
    /// ----------
    /// x : float or list of float
    ///     Point(s) at which to evaluate the interpolation.
    ///
    /// Returns
    /// -------
    /// float or list of float
    ///     Interpolated value(s) at the specified point(s).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the interpolator has not been fitted, or if the input
    ///     type is invalid.
    ///
    /// Examples
    /// --------
    /// >>> interp = LagrangeInterpolator()
    /// >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    /// >>> interp(1.5)          # single point  → 2.25
    /// >>> interp([0.5, 1.5])   # multiple points → [0.25, 2.25]
    pub fn __call__(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.fitted {
            return Err(PyValueError::new_err(
                "Interpolator not fitted. Call fit(x, y) first.",
            ));
        }

        // Single-point evaluation
        if let Ok(single_x) = x.extract::<f64>() {
            let result = barycentric_eval(
                &self.x_values,
                &self.y_values,
                &self.weights,
                single_x,
            );
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        // Multi-point evaluation
        if let Ok(x_list) = x.extract::<Vec<f64>>() {
            let results: Vec<f64> = x_list
                .iter()
                .map(|&xi| {
                    barycentric_eval(
                        &self.x_values,
                        &self.y_values,
                        &self.weights,
                        xi,
                    )
                })
                .collect();
            return Ok(results.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err(
            "Input must be a float or a list of floats",
        ))
    }

    /// String representation of the interpolator.
    ///
    /// Returns
    /// -------
    /// str
    ///     Description of the interpolator state.
    pub fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "LargangeInterpolator(barycentric, fitted with {} points)",
                self.x_values.len()
            )
        } else {
            "LagrangeInterpolator(barycentric, not fitted)".to_string()
        }
    }
}