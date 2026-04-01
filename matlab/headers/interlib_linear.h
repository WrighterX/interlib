#ifndef INTERLIB_LINEAR_H
#define INTERLIB_LINEAR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* interlib_linear_create(void);
void interlib_linear_destroy(void* handle);
int interlib_linear_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_linear_eval(void* handle, double x, double* out_value);
int interlib_linear_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_linear_last_error(char* buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
