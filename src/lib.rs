use pyo3::prelude::*;

// Module declarations
mod lagrange;
mod newton;
mod linear;
mod quadratic;
mod cubic_spline;
mod hermite;
mod least_squares;
mod rbf;
mod chebyshev;

/// A high-performance Python interpolation library implemented in Rust.
///
/// Provides various interpolators like LinearInterpolator, QuadraticInterpolator,
/// RBFInterpolator, etc.
///
/// Example:
///     from interlib import LinearInterpolator
///     interp = LinearInterpolator()
///     interp.fit([0.0, 1.0], [0.0, 1.0])
///     print(interp(0.5))  # 0.5
#[pymodule]
#[pyo3(name = "interlib")]
fn interlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // all interpolation classes
    m.add_class::<lagrange::LagrangeInterpolator>()?;
    m.add_class::<newton::NewtonInterpolator>()?;
    m.add_class::<linear::LinearInterpolator>()?;
    m.add_class::<quadratic::QuadraticInterpolator>()?;
    m.add_class::<cubic_spline::CubicSplineInterpolator>()?;
    m.add_class::<hermite::HermiteInterpolator>()?;
    m.add_class::<least_squares::LeastSquaresInterpolator>()?;
    m.add_class::<rbf::RBFInterpolator>()?;
    m.add_class::<chebyshev::ChebyshevInterpolator>()?;
    Ok(())
}