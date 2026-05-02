mod core;
mod matlab;
#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

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
    m.add_class::<python::linear::LinearInterpolator>()?;
    m.add_class::<python::lagrange::LagrangeInterpolator>()?;
    m.add_class::<python::newton::NewtonInterpolator>()?;
    m.add_class::<python::quadratic::QuadraticInterpolator>()?;
    m.add_class::<python::cubic_spline::CubicSplineInterpolator>()?;
    m.add_class::<python::hermite::HermiteInterpolator>()?;
    m.add_class::<python::least_squares::LeastSquaresInterpolator>()?;
    m.add_class::<python::rbf::RBFInterpolator>()?;
    m.add_class::<python::chebyshev::ChebyshevInterpolator>()?;
    Ok(())
}
