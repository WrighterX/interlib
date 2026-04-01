#ifndef INTERLIB_CUBIC_SPLINE_H
#define INTERLIB_CUBIC_SPLINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* interlib_cubic_spline_create(void);
void interlib_cubic_spline_destroy(void* handle);
int interlib_cubic_spline_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_cubic_spline_eval(void* handle, double x, double* out_value);
int interlib_cubic_spline_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_last_error(char* buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
