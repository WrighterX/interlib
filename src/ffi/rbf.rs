use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;

use crate::ffi::{clear_last_error, fail, last_error_string, success, write_last_error};
use crate::rbf_core::{RBFCore, RBFKernel};

fn fail_ptr(message: impl Into<String>) -> *mut c_void {
    fail(message);
    ptr::null_mut()
}

fn core_from_handle(handle: *mut c_void) -> Result<*mut RBFCore, &'static str> {
    if handle.is_null() {
        return Err("rbf handle is null");
    }
    Ok(handle as *mut RBFCore)
}

fn ptr_to_vec(ptr: *const f64, len: usize) -> Result<Vec<f64>, &'static str> {
    if ptr.is_null() {
        return Err("pointer must not be null");
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

fn kernel_from_id(kernel: c_int) -> Result<RBFKernel, &'static str> {
    RBFKernel::from_id(kernel as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_create(kernel: c_int, epsilon: f64) -> *mut c_void {
    clear_last_error();
    let kernel = match kernel_from_id(kernel) {
        Ok(k) => k,
        Err(msg) => return fail_ptr(msg),
    };
    match RBFCore::new(kernel, epsilon) {
        Ok(core) => Box::into_raw(Box::new(core)) as *mut c_void,
        Err(message) => fail_ptr(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_destroy(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(handle as *mut RBFCore)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_fit(
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
    let x = match ptr_to_vec(x_ptr, x_len) {
        Ok(vec) => vec,
        Err(message) => return fail(message),
    };
    let y = match ptr_to_vec(y_ptr, y_len) {
        Ok(vec) => vec,
        Err(message) => return fail(message),
    };

    let core = unsafe { &mut *core };
    match core.fit(&x, &y) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_eval(handle: *mut c_void, x: f64, out_value: *mut f64) -> i32 {
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
            unsafe { *out_value = value };
            success()
        }
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_eval_many(
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
    let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, x_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, x_len) };
    let core = unsafe { &mut *core };
    match core.fill_many(x_slice, out_slice) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_count(handle: *mut c_void) -> usize {
    match core_from_handle(handle) {
        Ok(core) => {
            let core = unsafe { &*core };
            core.point_count()
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_weights(
    handle: *mut c_void,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    let core = match core_from_handle(handle) {
        Ok(core) => core,
        Err(message) => return fail(message),
    };
    if out_ptr.is_null() {
        return fail("out_ptr must be non-null");
    }
    let core = unsafe { &*core };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
    match core.fill_weights(out_slice) {
        Ok(()) => success(),
        Err(message) => fail(message),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn interlib_rbf_last_error(buffer: *mut c_char, buffer_len: usize) -> usize {
    let message = last_error_string();
    write_last_error(&message, buffer, buffer_len)
}
