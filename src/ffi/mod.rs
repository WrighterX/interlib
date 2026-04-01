use std::cell::RefCell;
use std::os::raw::c_char;
use std::ptr;

pub(crate) mod linear;
pub(crate) mod newton;
pub(crate) mod quadratic;
pub(crate) mod cubic_spline;
pub(crate) mod hermite;
pub(crate) mod lagrange;
pub(crate) mod least_squares;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(crate) fn set_last_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = message.into();
    });
}

pub(crate) fn clear_last_error() {
    set_last_error("");
}

pub(crate) fn last_error_string() -> String {
    LAST_ERROR.with(|slot| slot.borrow().clone())
}

pub(crate) fn write_last_error(message: &str, buffer: *mut c_char, buffer_len: usize) -> usize {
    let bytes = message.as_bytes();
    let required_len = bytes.len() + 1;
    if buffer.is_null() || buffer_len == 0 {
        return required_len;
    }

    let copy_len = bytes.len().min(buffer_len.saturating_sub(1));
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buffer, copy_len);
        *buffer.add(copy_len) = 0;
    }
    required_len
}

pub(crate) fn fail(message: impl Into<String>) -> i32 {
    set_last_error(message);
    -1
}

pub(crate) fn success() -> i32 {
    clear_last_error();
    0
}

/// Copy the last error message into a caller-provided buffer.
///
/// Returns the number of bytes required including the trailing nul byte.
/// This is a shared entry point for all interpolators.
#[unsafe(no_mangle)]
pub extern "C" fn interlib_last_error(buffer: *mut c_char, buffer_len: usize) -> usize {
    let message = last_error_string();
    write_last_error(&message, buffer, buffer_len)
}
