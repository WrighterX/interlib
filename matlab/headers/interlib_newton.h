#ifndef INTERLIB_NEWTON_H
#define INTERLIB_NEWTON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* interlib_newton_create(void);
void interlib_newton_destroy(void* handle);
int interlib_newton_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_newton_eval(void* handle, double x, double* out_value);
int interlib_newton_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_last_error(char* buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
