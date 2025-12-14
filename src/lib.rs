use pyo3::prelude::*;

// Module declarations
mod lagrange;
mod newton;
mod linear;
mod quadratic;
mod cubic_spline;
mod hermite;
mod least_squares;

/// Python module definition
#[pymodule]
fn interlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // all interpolation classes
    m.add_class::<lagrange::LagrangeInterpolator>()?;
    m.add_class::<newton::NewtonInterpolator>()?;
    m.add_class::<linear::LinearInterpolator>()?;
    m.add_class::<quadratic::QuadraticInterpolator>()?;
    m.add_class::<cubic_spline::CubicSplineInterpolator>()?;
    m.add_class::<hermite::HermiteInterpolator>()?;
    m.add_class::<least_squares::LeastSquaresInterpolator>()?;
    
    Ok(())
}