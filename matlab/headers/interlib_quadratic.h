#ifndef INTERLIB_QUADRATIC_H
#define INTERLIB_QUADRATIC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* interlib_quadratic_create(void);
void interlib_quadratic_destroy(void* handle);
int interlib_quadratic_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_quadratic_eval(void* handle, double x, double* out_value);
int interlib_quadratic_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_last_error(char* buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
