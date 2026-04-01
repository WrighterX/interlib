#ifndef INTERLIB_NATIVE_H
#define INTERLIB_NATIVE_H

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
size_t interlib_newton_last_error(char* buffer, size_t buffer_len);

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
size_t interlib_cubic_spline_last_error(char* buffer, size_t buffer_len);

void* interlib_hermite_create(void);
void interlib_hermite_destroy(void* handle);
int interlib_hermite_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len,
    const double* dy_ptr,
    size_t dy_len
);
int interlib_hermite_eval(void* handle, double x, double* out_value);
int interlib_hermite_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_hermite_last_error(char* buffer, size_t buffer_len);

void* interlib_lagrange_create(void);
void interlib_lagrange_destroy(void* handle);
int interlib_lagrange_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_lagrange_update_y(void* handle, const double* y_ptr, size_t y_len);
int interlib_lagrange_add_point(void* handle, double x, double y);
int interlib_lagrange_eval(void* handle, double x, double* out_value);
int interlib_lagrange_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_lagrange_last_error(char* buffer, size_t buffer_len);

void* interlib_least_squares_create(size_t degree);
void interlib_least_squares_destroy(void* handle);
int interlib_least_squares_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_least_squares_eval(void* handle, double x, double* out_value);
int interlib_least_squares_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_least_squares_last_error(char* buffer, size_t buffer_len);

void* interlib_rbf_create(int kernel, double epsilon);
void interlib_rbf_destroy(void* handle);
int interlib_rbf_fit(
    void* handle,
    const double* x_ptr,
    size_t x_len,
    const double* y_ptr,
    size_t y_len
);
int interlib_rbf_eval(void* handle, double x, double* out_value);
int interlib_rbf_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_rbf_last_error(char* buffer, size_t buffer_len);
size_t interlib_rbf_count(void* handle);
int interlib_rbf_weights(void* handle, double* out_ptr, size_t out_len);

void* interlib_chebyshev_create(size_t n_points, double x_min, double x_max, int use_clenshaw);
void interlib_chebyshev_destroy(void* handle);
int interlib_chebyshev_fit(void* handle, const double* y_ptr, size_t y_len);
int interlib_chebyshev_eval(void* handle, double x, double* out_value);
int interlib_chebyshev_eval_many(void* handle, const double* x_ptr, size_t x_len, double* out_ptr);
size_t interlib_chebyshev_last_error(char* buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
