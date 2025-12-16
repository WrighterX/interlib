"""
Case Study 1: Function Approximation
=====================================
This study compares interpolation methods on mathematical functions:
1. cos(x) - smooth periodic function
2. Runge's function - notorious for polynomial oscillations
"""

import time
import math
import numpy as np
from interlib import (
    LagrangeInterpolator,
    NewtonInterpolator,
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    ChebyshevInterpolator,
    RBFInterpolator,
    LeastSquaresInterpolator
)


def test_cosine_approximation():
    """Test interpolation on cos(x) over [0, 2π]"""
    print("=" * 70)
    print("FUNCTION APPROXIMATION: cos(x) on [0, 2π]")
    print("=" * 70)
    
    # Training data
    n_train = 8
    x_train = np.linspace(0, 2*math.pi, n_train)
    y_train = np.cos(x_train)
    
    # Test data
    n_test = 100
    x_test = np.linspace(0, 2*math.pi, n_test)
    y_true = np.cos(x_test)
    
    results = {}
    
    # 1. Lagrange
    print("\n1. Lagrange Interpolation")
    start = time.perf_counter()
    lag = LagrangeInterpolator()
    lag.fit(x_train.tolist(), y_train.tolist())
    y_lag = np.array([lag(float(x)) for x in x_test])
    time_lag = time.perf_counter() - start
    error_lag = np.mean(np.abs(y_lag - y_true))
    max_error_lag = np.max(np.abs(y_lag - y_true))
    print(f"   Time: {time_lag:.6f}s, Mean Error: {error_lag:.6e}, Max Error: {max_error_lag:.6e}")
    results['Lagrange'] = (y_lag, time_lag, error_lag, max_error_lag)
    
    # 2. Newton
    print("\n2. Newton Interpolation")
    start = time.perf_counter()
    newton = NewtonInterpolator()
    newton.fit(x_train.tolist(), y_train.tolist())
    y_newton = np.array([newton(float(x)) for x in x_test])
    time_newton = time.perf_counter() - start
    error_newton = np.mean(np.abs(y_newton - y_true))
    max_error_newton = np.max(np.abs(y_newton - y_true))
    print(f"   Time: {time_newton:.6f}s, Mean Error: {error_newton:.6e}, Max Error: {max_error_newton:.6e}")
    results['Newton'] = (y_newton, time_newton, error_newton, max_error_newton)
    
    # 3. Cubic Spline
    print("\n3. Cubic Spline Interpolation")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(x_train.tolist(), y_train.tolist())
    y_spline = np.array([spline(float(x)) for x in x_test])
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(y_spline - y_true))
    max_error_spline = np.max(np.abs(y_spline - y_true))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.6e}, Max Error: {max_error_spline:.6e}")
    results['Cubic Spline'] = (y_spline, time_spline, error_spline, max_error_spline)
    
    # 4. Chebyshev
    print("\n4. Chebyshev Interpolation")
    start = time.perf_counter()
    cheb = ChebyshevInterpolator(n_points=n_train, x_min=0.0, x_max=2*math.pi)
    nodes = cheb.get_nodes()
    y_nodes = [math.cos(x) for x in nodes]
    cheb.fit(y_nodes)
    y_cheb = np.array([cheb(float(x)) for x in x_test])
    time_cheb = time.perf_counter() - start
    error_cheb = np.mean(np.abs(y_cheb - y_true))
    max_error_cheb = np.max(np.abs(y_cheb - y_true))
    print(f"   Time: {time_cheb:.6f}s, Mean Error: {error_cheb:.6e}, Max Error: {max_error_cheb:.6e}")
    results['Chebyshev'] = (y_cheb, time_cheb, error_cheb, max_error_cheb)
    
    # 5. RBF (Gaussian)
    print("\n5. RBF Interpolation (Gaussian)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="gaussian", epsilon=1.0)
    rbf.fit(x_train.tolist(), y_train.tolist())
    y_rbf = np.array([rbf(float(x)) for x in x_test])
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(y_rbf - y_true))
    max_error_rbf = np.max(np.abs(y_rbf - y_true))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.6e}, Max Error: {max_error_rbf:.6e}")
    results['RBF'] = (y_rbf, time_rbf, error_rbf, max_error_rbf)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: cos(x) Approximation")
    print("=" * 70)
    print(f"{'Method':<20} {'Mean Error':>15} {'Max Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (_, t, err, max_err) in results.items():
        print(f"{name:<20} {err:>15.6e} {max_err:>15.6e} {t*1000:>12.3f}")
    
    return results


def test_runge_function():
    """Test interpolation on Runge's function: 1/(1+25x²)"""
    print("\n\n" + "=" * 70)
    print("FUNCTION APPROXIMATION: Runge's Function 1/(1+25x²) on [-1, 1]")
    print("=" * 70)
    print("Note: This function is famous for causing oscillations with")
    print("      high-degree polynomial interpolation at uniform points.")
    
    # Training data - uniform points (problematic for polynomial interpolation)
    n_train = 11
    x_train = np.linspace(-1, 1, n_train)
    y_train = 1.0 / (1.0 + 25.0 * x_train**2)
    
    # Test data
    n_test = 200
    x_test = np.linspace(-1, 1, n_test)
    y_true = 1.0 / (1.0 + 25.0 * x_test**2)
    
    results = {}
    
    # 1. Lagrange (will show oscillations)
    print("\n1. Lagrange Interpolation (uniform points)")
    start = time.perf_counter()
    lag = LagrangeInterpolator()
    lag.fit(x_train.tolist(), y_train.tolist())
    y_lag = np.array([lag(float(x)) for x in x_test])
    time_lag = time.perf_counter() - start
    error_lag = np.mean(np.abs(y_lag - y_true))
    max_error_lag = np.max(np.abs(y_lag - y_true))
    print(f"   Time: {time_lag:.6f}s, Mean Error: {error_lag:.6e}, Max Error: {max_error_lag:.6e}")
    print(f"   WARNING: High max error indicates Runge's phenomenon!")
    results['Lagrange (uniform)'] = (y_lag, time_lag, error_lag, max_error_lag)
    
    # 2. Cubic Spline (should handle better)
    print("\n2. Cubic Spline Interpolation")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(x_train.tolist(), y_train.tolist())
    y_spline = np.array([spline(float(x)) for x in x_test])
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(y_spline - y_true))
    max_error_spline = np.max(np.abs(y_spline - y_true))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.6e}, Max Error: {max_error_spline:.6e}")
    results['Cubic Spline'] = (y_spline, time_spline, error_spline, max_error_spline)
    
    # 3. Chebyshev (optimal for this problem!)
    print("\n3. Chebyshev Interpolation (optimal node placement)")
    start = time.perf_counter()
    cheb = ChebyshevInterpolator(n_points=n_train, x_min=-1.0, x_max=1.0)
    nodes = cheb.get_nodes()
    y_nodes = [1.0 / (1.0 + 25.0 * x**2) for x in nodes]
    cheb.fit(y_nodes)
    y_cheb = np.array([cheb(float(x)) for x in x_test])
    time_cheb = time.perf_counter() - start
    error_cheb = np.mean(np.abs(y_cheb - y_true))
    max_error_cheb = np.max(np.abs(y_cheb - y_true))
    print(f"   Time: {time_cheb:.6f}s, Mean Error: {error_cheb:.6e}, Max Error: {max_error_cheb:.6e}")
    print(f"   SUCCESS: Chebyshev nodes avoid Runge's phenomenon!")
    results['Chebyshev'] = (y_cheb, time_cheb, error_cheb, max_error_cheb)
    
    # 4. RBF
    print("\n4. RBF Interpolation (Thin Plate Spline)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="thin_plate_spline", epsilon=1.0)
    rbf.fit(x_train.tolist(), y_train.tolist())
    y_rbf = np.array([rbf(float(x)) for x in x_test])
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(y_rbf - y_true))
    max_error_rbf = np.max(np.abs(y_rbf - y_true))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.6e}, Max Error: {max_error_rbf:.6e}")
    results['RBF'] = (y_rbf, time_rbf, error_rbf, max_error_rbf)
    
    # 5. Least Squares (lower degree to avoid oscillations)
    print("\n5. Least Squares (degree 6, smoother fit)")
    start = time.perf_counter()
    ls = LeastSquaresInterpolator(degree=6)
    ls.fit(x_train.tolist(), y_train.tolist())
    y_ls = np.array([ls(float(x)) for x in x_test])
    time_ls = time.perf_counter() - start
    error_ls = np.mean(np.abs(y_ls - y_true))
    max_error_ls = np.max(np.abs(y_ls - y_true))
    r_squared = ls.r_squared()
    print(f"   Time: {time_ls:.6f}s, Mean Error: {error_ls:.6e}, Max Error: {max_error_ls:.6e}")
    print(f"   R² = {r_squared:.6f}")
    results['Least Squares'] = (y_ls, time_ls, error_ls, max_error_ls)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Runge's Function Approximation")
    print("=" * 70)
    print(f"{'Method':<25} {'Mean Error':>15} {'Max Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (_, t, err, max_err) in results.items():
        print(f"{name:<25} {err:>15.6e} {max_err:>15.6e} {t*1000:>12.3f}")
    
    print("\nKEY INSIGHT:")
    print("  - Lagrange with uniform points shows large oscillations (Runge's phenomenon)")
    print("  - Chebyshev nodes dramatically reduce the error")
    print("  - Cubic spline provides local control, avoiding global oscillations")
    print("  - RBF and Least Squares offer smooth approximations")
    
    return results


def main():
    """Run all function approximation tests"""
    print("\n" + "#" * 70)
    print("# CASE STUDY 1: FUNCTION APPROXIMATION")
    print("#" * 70)
    
    total_start = time.perf_counter()
    
    # Test 1: cos(x)
    cosine_results = test_cosine_approximation()
    
    # Test 2: Runge's function
    runge_results = test_runge_function()
    
    total_time = time.perf_counter() - total_start
    
    print("\n" + "=" * 70)
    print(f"Total execution time: {total_time:.3f} seconds")
    print("=" * 70)
    
    print("\n\nCONCLUSIONS:")
    print("-" * 70)
    print("1. For smooth periodic functions (cos):")
    print("   - All methods perform well")
    print("   - Chebyshev and Cubic Spline show best accuracy")
    print("   - Newton and Lagrange are mathematically equivalent")
    print("\n2. For Runge's function:")
    print("   - Uniform polynomial interpolation (Lagrange) fails dramatically")
    print("   - Chebyshev nodes solve the oscillation problem")
    print("   - Local methods (Cubic Spline) avoid global instability")
    print("   - Lower-degree approximations (Least Squares) can be more stable")


if __name__ == "__main__":
    main()