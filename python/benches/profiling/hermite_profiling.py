"""
Hermite Interpolator Profiling Script

Measures fit() and __call__() separately to identify bottlenecks.
Helps answer:
- Where is time actually spent? (fit vs eval)
- How does it scale with data size?
- What's the fit/eval ratio?
"""

import time
import numpy as np
from typing import Tuple, List
import sys

from interlib import HermiteInterpolator

def generate_hermite_data(n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for Hermite interpolation."""
    x = np.linspace(0, 10, n_points)
    y = np.sin(x)  # Function: y = sin(x)
    dy = np.cos(x)  # Derivative: dy/dx = cos(x)
    x_test = np.linspace(0, 10, n_points * 10)  # Evaluation points
    return x, y, dy, x_test

def profile_hermite_fit(n_points: int, n_runs: int = 5) -> float:
    """
    Profile the fit() method.

    Returns:
        Average fit time in milliseconds
    """
    x, y, dy, _ = generate_hermite_data(n_points)

    times = []
    for _ in range(n_runs):
        interp = HermiteInterpolator()

        start = time.perf_counter()
        interp.fit(x.tolist(), y.tolist(), dy.tolist())
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)  # Convert to ms

    return np.mean(times)

def profile_hermite_eval(n_points: int, n_runs: int = 5) -> Tuple[float, float]:
    """
    Profile the __call__() method for both single point and batch evaluation.

    Returns:
        Tuple of (single_point_time_ms, batch_time_ms)
    """
    x, y, dy, x_test = generate_hermite_data(n_points)

    interp = HermiteInterpolator()
    interp.fit(x.tolist(), y.tolist(), dy.tolist())

    # Single point evaluation
    single_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = interp(5.0)
        elapsed = time.perf_counter() - start
        single_times.append(elapsed * 1000)

    # Batch evaluation (NumPy array - uses Idea 6)
    batch_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = interp(x_test)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed * 1000)

    return np.mean(single_times), np.mean(batch_times)

def profile_hermite_total(n_points: int, n_runs: int = 5) -> Tuple[float, float, float]:
    """
    Profile total time including fit and eval.

    Returns:
        Tuple of (fit_time, eval_time, total_time) in milliseconds
    """
    x, y, dy, x_test = generate_hermite_data(n_points)

    times = []
    eval_times = []
    fit_times = []

    for _ in range(n_runs):
        interp = HermiteInterpolator()

        # Measure fit
        start = time.perf_counter()
        interp.fit(x.tolist(), y.tolist(), dy.tolist())
        fit_time = (time.perf_counter() - start) * 1000
        fit_times.append(fit_time)

        # Measure eval (batch with NumPy)
        start = time.perf_counter()
        _ = interp(x_test)
        eval_time = (time.perf_counter() - start) * 1000
        eval_times.append(eval_time)

        times.append(fit_time + eval_time)

    return np.mean(fit_times), np.mean(eval_times), np.mean(times)

def main():
    print("=" * 80)
    print("HERMITE INTERPOLATOR PROFILING ANALYSIS")
    print("=" * 80)
    print()

    sizes = [10, 50, 100, 200, 500]

    print("Phase 1: Fit Time Analysis (O(n²) expected)")
    print("-" * 80)
    print(f"{'Data Points':<15} {'Fit Time (ms)':<20} {'Notes':<45}")
    print("-" * 80)

    fit_times = {}
    for size in sizes:
        fit_time = profile_hermite_fit(size, n_runs=3)
        fit_times[size] = fit_time
        print(f"{size:<15} {fit_time:<20.4f} O(n²) - divided differences table")

    print()
    print("Phase 2: Evaluation Time Analysis (Single vs Batch with NumPy)")
    print("-" * 80)
    print(f"{'Data Points':<15} {'Single (ms)':<20} {'Batch (ms)':<20} {'Batch/point (μs)':<20}")
    print("-" * 80)

    eval_times_single = {}
    eval_times_batch = {}

    for size in sizes:
        single, batch = profile_hermite_eval(size, n_runs=3)
        eval_times_single[size] = single
        eval_times_batch[size] = batch
        batch_per_point = (batch * 1000) / (size * 10)  # microseconds per point
        print(f"{size:<15} {single:<20.6f} {batch:<20.4f} {batch_per_point:<20.2f}")

    print()
    print("Phase 3: Total Time Breakdown (Fit vs Eval)")
    print("-" * 80)
    print(f"{'Data Points':<15} {'Fit (ms)':<20} {'Eval (ms)':<20} {'Total (ms)':<20} {'Fit %':<15}")
    print("-" * 80)

    for size in sizes:
        fit_time, eval_time, total_time = profile_hermite_total(size, n_runs=3)
        fit_percent = (fit_time / total_time) * 100 if total_time > 0 else 0
        print(f"{size:<15} {fit_time:<20.4f} {eval_time:<20.4f} {total_time:<20.4f} {fit_percent:<14.1f}%")

    print()
    print("Phase 4: Scaling Analysis")
    print("-" * 80)

    # Analyze fit time scaling
    print("\nFit Time Scaling (expect O(n²)):")
    for i in range(1, len(sizes)):
        size_ratio = sizes[i] / sizes[i-1]
        time_ratio = fit_times[sizes[i]] / fit_times[sizes[i-1]]
        expected_ratio = size_ratio ** 2
        print(f"  {sizes[i-1]}→{sizes[i]}: size×{size_ratio:.1f}, time×{time_ratio:.2f} (expected ×{expected_ratio:.1f})")

    print("\nBatch Eval Time Scaling (expect O(n) roughly linear with eval points):")
    for i in range(1, len(sizes)):
        size_ratio = sizes[i] / sizes[i-1]
        time_ratio = eval_times_batch[sizes[i]] / eval_times_batch[sizes[i-1]]
        # Eval points = n * 10, so ratio should be roughly size_ratio
        print(f"  {sizes[i-1]}→{sizes[i]}: size×{size_ratio:.1f}, time×{time_ratio:.2f} (expected ≈×{size_ratio:.1f})")

    print()
    print("=" * 80)
    print("PROFILING SUMMARY & INSIGHTS")
    print("=" * 80)
    print()

    # Find dominant bottleneck at 500 points
    fit_500, eval_500, total_500 = profile_hermite_total(500, n_runs=3)
    fit_percent_500 = (fit_500 / total_500) * 100
    eval_percent_500 = (eval_500 / total_500) * 100

    print(f"At 500 data points:")
    print(f"  • Fit time: {fit_500:.2f} ms ({fit_percent_500:.1f}% of total)")
    print(f"  • Eval time: {eval_500:.2f} ms ({eval_percent_500:.1f}% of total)")
    print(f"  • Total: {total_500:.2f} ms")
    print()

    if fit_percent_500 > 80:
        print("[FOUND] BOTTLENECK IDENTIFIED: fit() dominates (>80%)")
        print("  -> Optimization targets:")
        print("     1. Algorithm improvements (different divided differences method)")
        print("     2. SIMD vectorization for matrix operations")
        print("     3. Parallel fit (if possible, given dependencies)")
        print("     4. Use BLAS/LAPACK library")
    elif eval_percent_500 > 50:
        print("[FOUND] BOTTLENECK IDENTIFIED: eval() is significant (>50%)")
        print("  -> Idea 6 (NumPy arrays) should help")
        print("  -> Further optimizations: SIMD, cache tuning")
    else:
        print("[INFO] BALANCED: Both fit and eval contribute significantly")
        print("  -> Optimize both areas")

    print()
    print("Next steps:")
    print("  1. Compare these results with scipy.interpolate.BarycentricInterpolator")
    print("  2. Use rust-profiling skill for CPU flamegraph analysis")
    print("  3. Check allocation patterns with memory profilers")
    print("  4. Apply targeted optimizations based on findings")
    print()

if __name__ == "__main__":
    main()
