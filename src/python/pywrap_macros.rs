use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::core_trait::InterpolationCore;

/// Shared implementation of the `__call__` method for all interpolators.
///
/// Accepts a single float, a list of floats, or a NumPy array.
/// Returns a Python float for scalar input, or a Python list/array otherwise.
pub fn py_call_impl(
    core: &impl InterpolationCore,
    py: Python<'_>,
    x: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if let Ok(single) = x.extract::<f64>() {
        let value = core
            .evaluate_single(single)
            .map_err(PyValueError::new_err)?;
        return Ok(value.into_pyobject(py)?.into_any().unbind());
    }

    if let Ok(arr) = x.extract::<PyReadonlyArray1<f64>>() {
        let x_slice = arr.as_slice()?;
        let result_array = unsafe { PyArray1::<f64>::new(py, [x_slice.len()], false) };
        let result_slice = unsafe { result_array.as_slice_mut()? };
        core.fill_many(x_slice, result_slice)
            .map_err(PyValueError::new_err)?;
        return Ok(result_array.into_any().unbind());
    }

    if let Ok(x_list) = x.extract::<Vec<f64>>() {
        let results = core.evaluate_many(&x_list).map_err(PyValueError::new_err)?;
        return Ok(results.into_pyobject(py)?.into_any().unbind());
    }

    Err(PyValueError::new_err(
        "Input must be a float, list of floats, or NumPy array",
    ))
}
