use std::ffi::c_void;
use std::os::raw::c_char;

use crate::chebyshev_core::ChebyshevCore;
use crate::ffi::{clear_last_error, fail, last_error_string, success, write_last_error};

fn core_from_handle(handle: *mut c_void) -> Result<*mut ChebyshevCore, &'static str> {
    if handle.is_null() {
        return Err("chebyshev handle is null");
    }
    Ok(handle as *mut ChebyshevCore)
}

fn ptr_to_slice(ptr: *const f64, len: usize) -> Result<&'static [f64], &'static str> {
    if ptr.is_null() {
        return Err("pointer must not be null");
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn ptr_to_mut_slice(ptr: *mut f64, len: usize) -> Result<&'static mut [f64], &'static str> {
    if ptr.is_null() {
        return Err("pointer must not be null");
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_chebyshev_create(
    n_points: usize,
    x_min: f64,
    x_max: f64,
    use_clenshaw: bool,
) -> *mut c_void {
    clear_last_error();
    match ChebyshevCore::new(n_points, x_min, x_max, use_clenshaw) {
        Ok(core) => Box::into_raw(Box::new(core)) as *mut c_void,
        Err(message) => {
            fail(message);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_chebyshev_destroy(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(handle as *mut ChebyshevCore)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_chebyshev_fit(
    handle: *mut c_void,
    y_ptr: *const f64,
    y_len: usize,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    let y_slice = match ptr_to_slice(y_ptr, y_len) {
        Ok(slice) => slice,
        Err(message) => return fail(message),
    };
    let core = unsafe { &mut *core };
    match core.fit(y_slice) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_chebyshev_eval(handle: *mut c_void, x: f64, out_value: *mut f64) -> i32 {
    if out_value.is_null() {
        return fail("out_value must be non-null");
    }
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    let core = unsafe { &mut *core };
    match core.evaluate_single(x) {
        Ok(value) => {
            unsafe { *out_value = value };
            success()
        }
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_chebyshev_eval_many(
    handle: *mut c_void,
    x_ptr: *const f64,
    x_len: usize,
    out_ptr: *mut f64,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    let x_slice = match ptr_to_slice(x_ptr, x_len) {
        Ok(slice) => slice,
        Err(message) => return fail(message),
    };
    let out_slice = match ptr_to_mut_slice(out_ptr, x_len) {
        Ok(slice) => slice,
        Err(message) => return fail(message),
    };
    let core = unsafe { &mut *core };
    match core.fill_many(x_slice, out_slice) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_chebyshev_last_error(buffer: *mut c_char, buffer_len: usize) -> usize {
    let message = last_error_string();
    write_last_error(&message, buffer, buffer_len)
}
