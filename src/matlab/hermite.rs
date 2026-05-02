use crate::core::core_trait::InterpolationCore;
use std::ffi::c_void;
use std::os::raw::c_char;

use crate::matlab::{clear_last_error, fail, last_error_string, success, write_last_error};
use crate::core::hermite_core::HermiteCore;

fn core_from_handle(handle: *mut c_void) -> Result<*mut HermiteCore, &'static str> {
    if handle.is_null() {
        return Err("hermite handle is null");
    }
    Ok(handle as *mut HermiteCore)
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_hermite_create() -> *mut c_void {
    clear_last_error();
    Box::into_raw(Box::new(HermiteCore::new())) as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_hermite_destroy(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut HermiteCore));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_hermite_fit(
    handle: *mut c_void,
    x_ptr: *const f64,
    x_len: usize,
    y_ptr: *const f64,
    y_len: usize,
    dy_ptr: *const f64,
    dy_len: usize,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    if x_ptr.is_null() || y_ptr.is_null() || dy_ptr.is_null() {
        return fail("x_ptr, y_ptr, and dy_ptr must be non-null");
    }
    if x_len != y_len || x_len != dy_len {
        return fail("x, y, and dy must all have the same length");
    }
    if x_len == 0 {
        return fail("x, y, and dy cannot be empty");
    }

    let x = unsafe { std::slice::from_raw_parts(x_ptr, x_len) };
    let y = unsafe { std::slice::from_raw_parts(y_ptr, y_len) };
    let dy = unsafe { std::slice::from_raw_parts(dy_ptr, dy_len) };

    let core = unsafe { &mut *core };
    match core.fit(x.to_vec(), y.to_vec(), dy.to_vec()) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_hermite_eval(handle: *mut c_void, x: f64, out_value: *mut f64) -> i32 {
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

#[unsafe(no_mangle)]
pub extern "C" fn interlib_hermite_eval_many(
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

#[unsafe(no_mangle)]
pub extern "C" fn interlib_hermite_last_error(buffer: *mut c_char, buffer_len: usize) -> usize {
    let message = last_error_string();
    write_last_error(&message, buffer, buffer_len)
}
