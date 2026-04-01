use std::ffi::c_void;
use std::os::raw::c_char;

use crate::cubic_spline_core::CubicSplineCore;
use crate::ffi::{clear_last_error, fail, success, last_error_string, write_last_error};

fn core_from_handle(handle: *mut c_void) -> Result<*mut CubicSplineCore, &'static str> {
    if handle.is_null() {
        return Err("cubic_spline handle is null");
    }
    Ok(handle as *mut CubicSplineCore)
}

/// Create a new cubic spline interpolator handle for C consumers.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_cubic_spline_create() -> *mut c_void {
    clear_last_error();
    Box::into_raw(Box::new(CubicSplineCore::new())) as *mut c_void
}

/// Destroy a cubic spline interpolator handle created by `interlib_cubic_spline_create`.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_cubic_spline_destroy(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle as *mut CubicSplineCore));
    }
}

/// Fit the cubic spline interpolator with x/y data.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_cubic_spline_fit(
    handle: *mut c_void,
    x_ptr: *const f64,
    x_len: usize,
    y_ptr: *const f64,
    y_len: usize,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    if x_ptr.is_null() || y_ptr.is_null() {
        return fail("x_ptr and y_ptr must be non-null");
    }
    if x_len != y_len {
        return fail("x and y must have the same length");
    }
    if x_len < 2 {
        return fail("Cubic spline interpolation requires at least 2 data points");
    }

    let x = unsafe { std::slice::from_raw_parts(x_ptr, x_len) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, y_len) };

    let core = unsafe { &mut *core };
    match core.fit(x.to_vec(), y.to_vec()) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

/// Evaluate the cubic spline interpolator at a single point.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_cubic_spline_eval(
    handle: *mut c_void,
    x: f64,
    out_value: *mut f64,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    if out_value.is_null() {
        return fail("out_value must be non-null");
    }

    let core = unsafe { &mut *core };
    match core.evaluate_single(x) {
        Ok(value) => {
            unsafe {
                *out_value = value;
            }
            success()
        }
        Err(message) => fail(message),
    }
}

/// Evaluate the cubic spline interpolator for a contiguous array of points.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_cubic_spline_eval_many(
    handle: *mut c_void,
    x_ptr: *const f64,
    x_len: usize,
    out_ptr: *mut f64,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    if x_ptr.is_null() || out_ptr.is_null() {
        return fail("x_ptr and out_ptr must be non-null");
    }
    if x_len == 0 {
        clear_last_error();
        return 0;
    }

    let x = unsafe { std::slice::from_raw_parts(x_ptr, x_len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, x_len) };

    let core = unsafe { &mut *core };
    match core.fill_many(x, out) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

/// Copy the last error message into a caller-provided buffer.
///
/// Returns the number of bytes required including the trailing nul byte.
/// DEPRECATED: Use interlib_last_error instead.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_cubic_spline_last_error(buffer: *mut c_char, buffer_len: usize) -> usize {
    let message = last_error_string();
    write_last_error(&message, buffer, buffer_len)
}

// Keep the cubic spline entry points reachable in the cdylib so the linker
// does not discard them from the MATLAB-targeted shared library.
#[used]
static KEEP_CUBIC_SPLINE_CREATE: extern "C" fn() -> *mut c_void = interlib_cubic_spline_create;
#[used]
static KEEP_CUBIC_SPLINE_DESTROY: extern "C" fn(*mut c_void) = interlib_cubic_spline_destroy;
#[used]
static KEEP_CUBIC_SPLINE_FIT: extern "C" fn(*mut c_void, *const f64, usize, *const f64, usize) -> i32 =
    interlib_cubic_spline_fit;
#[used]
static KEEP_CUBIC_SPLINE_EVAL: extern "C" fn(*mut c_void, f64, *mut f64) -> i32 =
    interlib_cubic_spline_eval;
#[used]
static KEEP_CUBIC_SPLINE_EVAL_MANY: extern "C" fn(*mut c_void, *const f64, usize, *mut f64) -> i32 =
    interlib_cubic_spline_eval_many;
#[used]
static KEEP_CUBIC_SPLINE_LAST_ERROR: extern "C" fn(*mut c_char, usize) -> usize =
    interlib_cubic_spline_last_error;
