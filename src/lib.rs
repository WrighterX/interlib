use pyo3::prelude::*;

// Module declarations
mod lagrange;
// Uncomment these as you implement them:
mod newton;
// mod linear;

/// Python module definition
#[pymodule]
fn interlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // all interpolation classes
    m.add_class::<lagrange::LagrangeInterpolator>()?;
    
    // Uncomment these as you add more methods:
    m.add_class::<newton::NewtonInterpolator>()?;
    // m.add_class::<linear::LinearInterpolator>()?;
    
    Ok(())
}