"""
Compare interlib Hermite Interpolator with scipy alternatives.

Compares:
1. interlib.HermiteInterpolator (Rust implementation)
2. scipy.interpolate.BarycentricInterpolator (NumPy-based)
3. scipy.interpolate.CubicSpline (NumPy-based)

Measures both fit and eval time to show where scipy excels and where we lag.
"""

import time
import numpy as np
from typing import Tuple
import sys

from interlib import HermiteInterpolator
from scipy.interpolate import BarycentricInterpolator, CubicSpline


def generate_test_data(n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for interpolation."""
    x = np.linspace(0, 10, n_points)
    y = np.sin(x)
    dy = np.cos(x)
    x_test = np.linspace(0, 10, n_points * 10)
    return x, y, dy, x_test


def benchmark_hermite(x: np.ndarray, y: np.ndarray, dy: np.ndarray,
                      x_test: np.ndarray, n_runs: int = 3) -> Tuple[float, float]:
    """Benchmark interlib HermiteInterpolator."""
    times_fit = []
    times_eval = []

    for _ in range(n_runs):
        interp = HermiteInterpolator()

        start = time.perf_counter()
        interp.fit(x.tolist(), y.tolist(), dy.tolist())
        fit_time = (time.perf_counter() - start) * 1000
        times_fit.append(fit_time)

        start = time.perf_counter()
        _ = interp(x_test)
        eval_time = (time.perf_counter() - start) * 1000
        times_eval.append(eval_time)

    return np.mean(times_fit), np.mean(times_eval)


def benchmark_barycentric(x: np.ndarray, y: np.ndarray,
                          x_test: np.ndarray, n_runs: int = 3) -> Tuple[float, float]:
    """Benchmark scipy BarycentricInterpolator."""
    times_fit = []
    times_eval = []

    for _ in range(n_runs):
        start = time.perf_counter()
        interp = BarycentricInterpolator(x, y)
        fit_time = (time.perf_counter() - start) * 1000
        times_fit.append(fit_time)

        start = time.perf_counter()
        _ = interp(x_test)
        eval_time = (time.perf_counter() - start) * 1000
        times_eval.append(eval_time)

    return np.mean(times_fit), np.mean(times_eval)


def benchmark_cubic_spline(x: np.ndarray, y: np.ndarray,
                           x_test: np.ndarray, n_runs: int = 3) -> Tuple[float, float]:
    """Benchmark scipy CubicSpline."""
    times_fit = []
    times_eval = []

    for _ in range(n_runs):
        start = time.perf_counter()
        interp = CubicSpline(x, y)
        fit_time = (time.perf_counter() - start) * 1000
        times_fit.append(fit_time)

        start = time.perf_counter()
        _ = interp(x_test)
        eval_time = (time.perf_counter() - start) * 1000
        times_eval.append(eval_time)

    return np.mean(times_fit), np.mean(times_eval)


def main():
    print("=" * 100)
    print("SCIPY COMPARISON: interlib vs scipy interpolators")
    print("=" * 100)
    print()

    sizes = [10, 50, 100, 200, 500]

    print("Method                          Data Points    Fit (ms)    Eval (ms)    Total (ms)   Speedup (Hermite vs scipy)")
    print("-" * 100)

    for size in sizes:
        x, y, dy, x_test = generate_test_data(size)

        # Hermite
        h_fit, h_eval = benchmark_hermite(x, y, dy, x_test, n_runs=3)
        h_total = h_fit + h_eval

        # Barycentric
        b_fit, b_eval = benchmark_barycentric(x, y, x_test, n_runs=3)
        b_total = b_fit + b_eval

        # CubicSpline
        cs_fit, cs_eval = benchmark_cubic_spline(x, y, x_test, n_runs=3)
        cs_total = cs_fit + cs_eval

        speedup_vs_bary = b_total / h_total if h_total > 0 else float('inf')
        speedup_vs_cs = cs_total / h_total if h_total > 0 else float('inf')

        print(f"{'interlib.Hermite':<30} {size:<14} {h_fit:<10.4f} {h_eval:<12.4f} {h_total:<12.4f} (baseline)")
        print(f"{'scipy.Barycentric':<30} {size:<14} {b_fit:<10.4f} {b_eval:<12.4f} {b_total:<12.4f} {speedup_vs_bary:.2f}x {'FASTER' if speedup_vs_bary > 1.0 else 'slower'}")
        print(f"{'scipy.CubicSpline':<30} {size:<14} {cs_fit:<10.4f} {cs_eval:<12.4f} {cs_total:<12.4f} {speedup_vs_cs:.2f}x {'FASTER' if speedup_vs_cs > 1.0 else 'slower'}")
        print()

    print("=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print()
    print("Key insights:")
    print("1. If Hermite is faster than scipy methods → Our optimization strategy worked! Keep focus on Idea 6 variants.")
    print("2. If scipy Barycentric is faster → Compare algorithm efficiency (Barycentric uses different method than Hermite)")
    print("3. If scipy CubicSpline is faster → Focus on O(n²) fit() optimization with SIMD or algorithm change")
    print("4. Eval time comparison → Shows effectiveness of Idea 6 (NumPy batching)")
    print()


if __name__ == "__main__":
    main()
