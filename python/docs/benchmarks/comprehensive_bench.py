"""
Comprehensive Benchmark Suite for interlib vs scipy.interpolate

This script performs extensive performance testing across all interpolation methods
with varying dataset sizes and provides detailed statistical analysis.
"""

import time
import numpy as np
from typing import Callable, Tuple, List, Dict
import sys

# Import scipy methods
from scipy.interpolate import (
    BarycentricInterpolator,
    CubicSpline,
    RBFInterpolator as ScipyRBFInterpolator,
    interp1d,
    PchipInterpolator
)

# Import interlib methods
from interlib import (
    LagrangeInterpolator,
    NewtonInterpolator,
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    HermiteInterpolator,
    LeastSquaresInterpolator,
    RBFInterpolator,
    ChebyshevInterpolator
)


def generate_test_data(n_points: int, func: str = "sin") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test data for benchmarking.

    Parameters
    ----------
    n_points : int
        Number of data points
    func : str
        Function type: "sin", "poly", or "exp"

    Returns
    -------
    x_train : np.ndarray
        Training x coordinates
    y_train : np.ndarray
        Training y coordinates
    x_test : np.ndarray
        Test points for evaluation
    """
    x_train = np.linspace(0, 10, n_points)

    if func == "sin":
        y_train = np.sin(x_train)
    elif func == "poly":
        y_train = x_train**2 - 2*x_train + 1
    elif func == "exp":
        y_train = np.exp(-x_train/5) * np.sin(x_train)
    else:
        raise ValueError(f"Unknown function: {func}")

    # Generate test points (interpolation points between training data)
    x_test = np.linspace(0, 10, n_points * 10)

    return x_train, y_train, x_test


def benchmark_method(
    fit_func: Callable,
    eval_func: Callable,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_runs: int = 5
) -> Tuple[float, float, float]:
    """
    Benchmark a single interpolation method.

    Parameters
    ----------
    fit_func : callable
        Function to fit the interpolator
    eval_func : callable
        Function to evaluate the interpolator
    x_train, y_train : np.ndarray
        Training data
    x_test : np.ndarray
        Test points
    n_runs : int
        Number of runs for averaging

    Returns
    -------
    fit_time : float
        Average fit time in seconds
    eval_time : float
        Average evaluation time in seconds
    total_time : float
        Average total time in seconds
    """
    fit_times = []
    eval_times = []

    for _ in range(n_runs):
        # Time fitting
        start = time.perf_counter()
        interpolator = fit_func(x_train, y_train)
        fit_time = time.perf_counter() - start
        fit_times.append(fit_time)

        # Time evaluation
        start = time.perf_counter()
        _ = eval_func(interpolator, x_test)
        eval_time = time.perf_counter() - start
        eval_times.append(eval_time)

    fit_mean = np.mean(fit_times)
    eval_mean = np.mean(eval_times)
    total_mean = fit_mean + eval_mean

    return fit_mean, eval_mean, total_mean


def run_lagrange_benchmark(x_train, y_train, x_test, n_runs=5):
    """Benchmark Lagrange interpolation."""
    # interlib
    def fit_interlib(x, y):
        interp = LagrangeInterpolator()
        interp.fit(x, y)
        return interp

    def eval_interlib(interp, x):
        return interp(x)

    t_fit_il, t_eval_il, t_total_il = benchmark_method(
        fit_interlib, eval_interlib, x_train, y_train, x_test, n_runs
    )

    # scipy
    def fit_scipy(x, y):
        return BarycentricInterpolator(x, y)

    def eval_scipy(interp, x):
        return interp(x)

    t_fit_sp, t_eval_sp, t_total_sp = benchmark_method(
        fit_scipy, eval_scipy, x_train, y_train, x_test, n_runs
    )

    return {
        "method": "Lagrange",
        "interlib_fit": t_fit_il,
        "interlib_eval": t_eval_il,
        "interlib_total": t_total_il,
        "scipy_fit": t_fit_sp,
        "scipy_eval": t_eval_sp,
        "scipy_total": t_total_sp,
        "speedup": t_total_sp / t_total_il if t_total_il > 0 else 0
    }


def run_cubicspline_benchmark(x_train, y_train, x_test, n_runs=5):
    """Benchmark Cubic Spline interpolation."""
    # interlib
    def fit_interlib(x, y):
        interp = CubicSplineInterpolator()
        interp.fit(x, y)
        return interp

    def eval_interlib(interp, x):
        return interp(x)

    t_fit_il, t_eval_il, t_total_il = benchmark_method(
        fit_interlib, eval_interlib, x_train, y_train, x_test, n_runs
    )

    # scipy
    def fit_scipy(x, y):
        return CubicSpline(x, y)

    def eval_scipy(interp, x):
        return interp(x)

    t_fit_sp, t_eval_sp, t_total_sp = benchmark_method(
        fit_scipy, eval_scipy, x_train, y_train, x_test, n_runs
    )

    return {
        "method": "CubicSpline",
        "interlib_fit": t_fit_il,
        "interlib_eval": t_eval_il,
        "interlib_total": t_total_il,
        "scipy_fit": t_fit_sp,
        "scipy_eval": t_eval_sp,
        "scipy_total": t_total_sp,
        "speedup": t_total_sp / t_total_il if t_total_il > 0 else 0
    }


def run_linear_benchmark(x_train, y_train, x_test, n_runs=5):
    """Benchmark Linear interpolation."""
    # interlib
    def fit_interlib(x, y):
        interp = LinearInterpolator()
        interp.fit(x, y)
        return interp

    def eval_interlib(interp, x):
        return interp(x)

    t_fit_il, t_eval_il, t_total_il = benchmark_method(
        fit_interlib, eval_interlib, x_train, y_train, x_test, n_runs
    )

    # scipy
    def fit_scipy(x, y):
        return interp1d(x, y, kind='linear')

    def eval_scipy(interp, x):
        return interp(x)

    t_fit_sp, t_eval_sp, t_total_sp = benchmark_method(
        fit_scipy, eval_scipy, x_train, y_train, x_test, n_runs
    )

    return {
        "method": "Linear",
        "interlib_fit": t_fit_il,
        "interlib_eval": t_eval_il,
        "interlib_total": t_total_il,
        "scipy_fit": t_fit_sp,
        "scipy_eval": t_eval_sp,
        "scipy_total": t_total_sp,
        "speedup": t_total_sp / t_total_il if t_total_il > 0 else 0
    }


def run_rbf_benchmark(x_train, y_train, x_test, n_runs=5):
    """Benchmark RBF interpolation."""
    # interlib
    def fit_interlib(x, y):
        interp = RBFInterpolator(kernel="gaussian", epsilon=1.0)
        interp.fit(x, y)
        return interp

    def eval_interlib(interp, x):
        return interp(x)

    t_fit_il, t_eval_il, t_total_il = benchmark_method(
        fit_interlib, eval_interlib, x_train, y_train, x_test, n_runs
    )

    # scipy (requires 2D input)
    x_train_2d = x_train.reshape(-1, 1)
    x_test_2d = x_test.reshape(-1, 1)

    def fit_scipy(x, y):
        return ScipyRBFInterpolator(x, y, kernel="gaussian", epsilon=1.0)

    def eval_scipy(interp, x):
        return interp(x).flatten()

    t_fit_sp, t_eval_sp, t_total_sp = benchmark_method(
        fit_scipy, eval_scipy, x_train_2d, y_train, x_test_2d, n_runs
    )

    return {
        "method": "RBF (Gaussian)",
        "interlib_fit": t_fit_il,
        "interlib_eval": t_eval_il,
        "interlib_total": t_total_il,
        "scipy_fit": t_fit_sp,
        "scipy_eval": t_eval_sp,
        "scipy_total": t_total_sp,
        "speedup": t_total_sp / t_total_il if t_total_il > 0 else 0
    }


def run_hermite_benchmark(x_train, y_train, x_test, n_runs=5):
    """Benchmark Hermite interpolation."""
    # Calculate derivatives for the test function (assuming sin)
    derivs = np.cos(x_train)

    # interlib
    def fit_interlib(x, y):
        interp = HermiteInterpolator()
        interp.fit(x, y, derivs)
        return interp

    def eval_interlib(interp, x):
        return interp(x)

    t_fit_il, t_eval_il, t_total_il = benchmark_method(
        fit_interlib, eval_interlib, x_train, y_train, x_test, n_runs
    )

    # scipy (using PCHIP as similar alternative)
    def fit_scipy(x, y):
        return PchipInterpolator(x, y)

    def eval_scipy(interp, x):
        return interp(x)

    t_fit_sp, t_eval_sp, t_total_sp = benchmark_method(
        fit_scipy, eval_scipy, x_train, y_train, x_test, n_runs
    )

    return {
        "method": "Hermite",
        "interlib_fit": t_fit_il,
        "interlib_eval": t_eval_il,
        "interlib_total": t_total_il,
        "scipy_fit": t_fit_sp,
        "scipy_eval": t_eval_sp,
        "scipy_total": t_total_sp,
        "speedup": t_total_sp / t_total_il if t_total_il > 0 else 0
    }


def print_results_table(results: List[Dict], n_points: int):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - Dataset Size: {n_points} points")
    print(f"{'='*80}\n")

    print(f"{'Method':<20} {'interlib (ms)':<15} {'scipy (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*80}")

    speedups = []
    for row in results:
        method = row['method']
        il_time = row['interlib_total'] * 1000  # Convert to ms
        sp_time = row['scipy_total'] * 1000
        speedup = row['speedup']
        speedups.append((speedup, method))

        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"{method:<20} {il_time:<15.4f} {sp_time:<15.4f} {speedup_str:<10}")

    print(f"{'-'*80}\n")

    # Summary statistics
    if speedups:
        avg_speedup = np.mean([s[0] for s in speedups])
        max_speedup, max_method = max(speedups)
        min_speedup, min_method = min(speedups)

        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Best Speedup:    {max_speedup:.2f}x ({max_method})")
        print(f"Worst Speedup:   {min_speedup:.2f}x ({min_method})")


def main():
    """Run comprehensive benchmarks."""
    print("\n" + "="*80)
    print("COMPREHENSIVE INTERLIB vs SCIPY BENCHMARK SUITE")
    print("="*80)

    # Test with different dataset sizes
    dataset_sizes = [10, 50, 100, 500]

    for n_points in dataset_sizes:
        x_train, y_train, x_test = generate_test_data(n_points, func="sin")

        results = []

        print(f"\nRunning benchmarks for {n_points} data points...")

        # Run benchmarks for each method
        try:
            results.append(run_lagrange_benchmark(x_train, y_train, x_test))
        except Exception as e:
            print(f"Lagrange benchmark failed: {e}")

        try:
            results.append(run_linear_benchmark(x_train, y_train, x_test))
        except Exception as e:
            print(f"Linear benchmark failed: {e}")

        try:
            results.append(run_cubicspline_benchmark(x_train, y_train, x_test))
        except Exception as e:
            print(f"CubicSpline benchmark failed: {e}")

        try:
            results.append(run_rbf_benchmark(x_train, y_train, x_test))
        except Exception as e:
            print(f"RBF benchmark failed: {e}")

        try:
            results.append(run_hermite_benchmark(x_train, y_train, x_test))
        except Exception as e:
            print(f"Hermite benchmark failed: {e}")

        # Print results
        if results:
            print_results_table(results, n_points)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
