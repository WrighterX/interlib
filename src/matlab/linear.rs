use crate::core::core_trait::InterpolationCore;
use std::ffi::c_void;
use std::os::raw::c_char;

use crate::matlab::{clear_last_error, fail, last_error_string, success, write_last_error};
use crate::core::linear_core::LinearCore;

fn core_from_handle(handle: *mut c_void) -> Result<*mut LinearCore, &'static str> {
    if handle.is_null() {
        return Err("linear handle is null");
    }
    Ok(handle as *mut LinearCore)
}

/// Create a new linear interpolator handle for C consumers.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_linear_create() -> *mut c_void {
    clear_last_error();
    Box::into_raw(Box::new(LinearCore::new())) as *mut c_void
}

/// Destroy a linear interpolator handle created by `interlib_linear_create`.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_linear_destroy(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle as *mut LinearCore));
    }
}

/// Fit the linear interpolator with x/y data.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_linear_fit(
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
    if x_len == 0 {
        return fail("x and y cannot be empty");
    }

    let x = unsafe { std::slice::from_raw_parts(x_ptr, x_len) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, y_len) };

    let core = unsafe { &mut *core };
    match core.fit(x.to_vec(), y.to_vec()) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

/// Evaluate the linear interpolator at a single point.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_linear_eval(handle: *mut c_void, x: f64, out_value: *mut f64) -> i32 {
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

/// Evaluate the linear interpolator for a contiguous array of points.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_linear_eval_many(
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
pub extern "C" fn interlib_linear_last_error(buffer: *mut c_char, buffer_len: usize) -> usize {
    let message = last_error_string();
    write_last_error(&message, buffer, buffer_len)
}

// Keep the linear entry points reachable in the cdylib so the linker
// does not discard them from the MATLAB-targeted shared library.
#[used]
static KEEP_LINEAR_CREATE: extern "C" fn() -> *mut c_void = interlib_linear_create;
#[used]
static KEEP_LINEAR_DESTROY: extern "C" fn(*mut c_void) = interlib_linear_destroy;
#[used]
static KEEP_LINEAR_FIT: extern "C" fn(*mut c_void, *const f64, usize, *const f64, usize) -> i32 =
    interlib_linear_fit;
#[used]
static KEEP_LINEAR_EVAL: extern "C" fn(*mut c_void, f64, *mut f64) -> i32 = interlib_linear_eval;
#[used]
static KEEP_LINEAR_EVAL_MANY: extern "C" fn(*mut c_void, *const f64, usize, *mut f64) -> i32 =
    interlib_linear_eval_many;
#[used]
static KEEP_LINEAR_LAST_ERROR: extern "C" fn(*mut c_char, usize) -> usize =
    interlib_linear_last_error;

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn ffi_round_trip_fit_and_eval() {
        let handle = interlib_linear_create();
        assert!(!handle.is_null());

        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 4.0];
        assert_eq!(
            interlib_linear_fit(handle, xs.as_ptr(), xs.len(), ys.as_ptr(), ys.len()),
            0
        );

        let mut value = 0.0;
        assert_eq!(interlib_linear_eval(handle, 0.5, &mut value), 0);
        assert!((value - 0.5).abs() < 1e-12);

        let mut out = [0.0; 3];
        assert_eq!(
            interlib_linear_eval_many(handle, xs.as_ptr(), xs.len(), out.as_mut_ptr()),
            0
        );
        assert_eq!(out, [0.0, 1.0, 4.0]);

        interlib_linear_destroy(handle);
    }

    #[test]
    fn ffi_reports_errors() {
        let status = interlib_linear_fit(
            std::ptr::null_mut(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
        );
        assert_eq!(status, -1);

        let mut buffer = [0i8; 64];
        let required = interlib_linear_last_error(buffer.as_mut_ptr(), buffer.len());
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str().unwrap();
        assert!(message.contains("linear handle is null"));
        assert!(required > 0);
    }
}
