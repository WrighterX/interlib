mod linear_core;
mod newton_core;
mod quadratic_core;
mod cubic_spline_core;
mod hermite_core;
mod ffi;
mod matlab;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
mod lagrange;
#[cfg(feature = "python")]
mod newton;
#[cfg(feature = "python")]
mod linear;
#[cfg(feature = "python")]
mod quadratic;
#[cfg(feature = "python")]
mod cubic_spline;
#[cfg(feature = "python")]
mod hermite;
#[cfg(feature = "python")]
mod least_squares;
#[cfg(feature = "python")]
mod rbf;
#[cfg(feature = "python")]
mod chebyshev;

/// A high-performance Python interpolation library implemented in Rust.
///
/// This module provides various interpolation algorithms optimized for performance.
/// It uses PyO3 for Python bindings and can handle both scalar and NumPy array inputs.
///
/// Example:
///     from interlib import LinearInterpolator
///     interp = LinearInterpolator()
///     interp.fit([0.0, 1.0], [0.0, 1.0])
///     print(interp(0.5))  # 0.5
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "interlib")]
fn interlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<linear::LinearInterpolator>()?;
    m.add_class::<lagrange::LagrangeInterpolator>()?;
    m.add_class::<newton::NewtonInterpolator>()?;
    m.add_class::<quadratic::QuadraticInterpolator>()?;
    m.add_class::<cubic_spline::CubicSplineInterpolator>()?;
    m.add_class::<hermite::HermiteInterpolator>()?;
    m.add_class::<least_squares::LeastSquaresInterpolator>()?;
    m.add_class::<rbf::RBFInterpolator>()?;
    m.add_class::<chebyshev::ChebyshevInterpolator>()?;
    Ok(())
}
