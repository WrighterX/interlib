"""
Correct method-to-method comparison: interlib.HermiteInterpolator vs scipy methods that use derivatives

Compares:
1. interlib.HermiteInterpolator - Global polynomial Hermite (uses y and dy)
2. scipy.interpolate.CubicHermiteSpline - Piecewise cubic Hermite (uses y and dy)
3. scipy.interpolate.KroghInterpolator - Global polynomial via divided differences (uses only y, no dy)

This is the CORRECT method-to-method comparison.
"""

import time
import numpy as np
from typing import Tuple
import sys

from interlib import HermiteInterpolator
from scipy.interpolate import CubicHermiteSpline, KroghInterpolator


def generate_test_data(n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for Hermite interpolation."""
    x = np.linspace(0, 10, n_points)
    y = np.sin(x)
    dy = np.cos(x)  # First derivative
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


def benchmark_scipy_hermite_spline(x: np.ndarray, y: np.ndarray, dy: np.ndarray,
                                   x_test: np.ndarray, n_runs: int = 3) -> Tuple[float, float]:
    """Benchmark scipy CubicHermiteSpline (piecewise cubic with derivatives)."""
    times_fit = []
    times_eval = []

    for _ in range(n_runs):
        start = time.perf_counter()
        interp = CubicHermiteSpline(x, y, dy)
        fit_time = (time.perf_counter() - start) * 1000
        times_fit.append(fit_time)

        start = time.perf_counter()
        _ = interp(x_test)
        eval_time = (time.perf_counter() - start) * 1000
        times_eval.append(eval_time)

    return np.mean(times_fit), np.mean(times_eval)


def benchmark_scipy_krogh(x: np.ndarray, y: np.ndarray,
                          x_test: np.ndarray, n_runs: int = 3) -> Tuple[float, float]:
    """Benchmark scipy KroghInterpolator (global polynomial, no derivatives)."""
    times_fit = []
    times_eval = []

    for _ in range(n_runs):
        start = time.perf_counter()
        interp = KroghInterpolator(x, y)
        fit_time = (time.perf_counter() - start) * 1000
        times_fit.append(fit_time)

        start = time.perf_counter()
        _ = interp(x_test)
        eval_time = (time.perf_counter() - start) * 1000
        times_eval.append(eval_time)

    return np.mean(times_fit), np.mean(times_eval)


def main():
    print("=" * 110)
    print("METHOD-TO-METHOD COMPARISON: Hermite Interpolators")
    print("=" * 110)
    print()
    print("Comparing interpolation methods that use derivative information:")
    print("  1. interlib.HermiteInterpolator - Global polynomial (uses y and dy)")
    print("  2. scipy.CubicHermiteSpline - Piecewise cubic (uses y and dy)")
    print("  3. scipy.KroghInterpolator - Global polynomial (uses y only, no dy)")
    print()

    sizes = [10, 50, 100, 200, 500]

    print("Method                                Data Pts   Fit (ms)   Eval (ms)  Total (ms)   vs Hermite")
    print("-" * 110)

    for size in sizes:
        x, y, dy, x_test = generate_test_data(size)

        # Our Hermite
        h_fit, h_eval = benchmark_hermite(x, y, dy, x_test, n_runs=3)
        h_total = h_fit + h_eval

        # Scipy CubicHermiteSpline (uses derivatives like our Hermite)
        hs_fit, hs_eval = benchmark_scipy_hermite_spline(x, y, dy, x_test, n_runs=3)
        hs_total = hs_fit + hs_eval

        # Scipy KroghInterpolator (global polynomial but no derivatives)
        k_fit, k_eval = benchmark_scipy_krogh(x, y, x_test, n_runs=3)
        k_total = k_fit + k_eval

        speedup_hs = h_total / hs_total if hs_total > 0 else float('inf')
        speedup_k = h_total / k_total if k_total > 0 else float('inf')

        print(f"{'interlib.HermiteInterpolator':<35} {size:<11} {h_fit:<10.4f} {h_eval:<10.4f} {h_total:<12.4f} (baseline)")
        print(f"{'scipy.CubicHermiteSpline (HERMITE)':<35} {size:<11} {hs_fit:<10.4f} {hs_eval:<10.4f} {hs_total:<12.4f} {speedup_hs:.2f}x {'FASTER' if speedup_hs > 1.0 else 'SLOWER'}")
        print(f"{'scipy.KroghInterpolator (global poly)':<35} {size:<11} {k_fit:<10.4f} {k_eval:<10.4f} {k_total:<12.4f} {speedup_k:.2f}x {'FASTER' if speedup_k > 1.0 else 'SLOWER'}")
        print()

    print("=" * 110)
    print("ANALYSIS")
    print("=" * 110)
    print()
    print("Key observations:")
    print("1. CubicHermiteSpline vs HermiteInterpolator:")
    print("   - Both use derivatives (true Hermite methods)")
    print("   - CubicHermiteSpline is PIECEWISE (local cubic segments)")
    print("   - Our HermiteInterpolator is GLOBAL (single polynomial)")
    print()
    print("2. KroghInterpolator vs HermiteInterpolator:")
    print("   - Both are GLOBAL polynomial (similar algorithm base)")
    print("   - KroghInterpolator ignores derivatives (simpler problem)")
    print("   - Our HermiteInterpolator uses derivatives (harder problem)")
    print()
    print("3. Performance interpretation:")
    print("   - If we're faster than CubicHermiteSpline: Good! Global > Piecewise for this problem")
    print("   - If we're slower than KroghInterpolator: Expected (Hermite adds derivative constraint)")
    print("   - If we're much slower than Krogh: Possible optimization opportunity")
    print()


if __name__ == "__main__":
    main()
