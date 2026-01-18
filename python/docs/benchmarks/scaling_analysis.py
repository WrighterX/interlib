"""
Dimensional Scaling Analysis for interlib

This script measures execution time as the number of data points grows,
allowing verification of algorithmic complexity (O(n), O(n²), etc.).
"""

import time
import numpy as np
from typing import Callable, List, Tuple
import sys

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


def measure_scaling(
    interpolator_class,
    data_sizes: List[int],
    n_runs: int = 3,
    **kwargs
) -> Tuple[List[int], List[float], List[float]]:
    """
    Measure how interpolation time scales with dataset size.

    Parameters
    ----------
    interpolator_class : class
        The interpolator class to test
    data_sizes : list of int
        List of dataset sizes to test
    n_runs : int
        Number of runs per size for averaging
    **kwargs : dict
        Additional arguments to pass to interpolator constructor

    Returns
    -------
    sizes : list of int
        Dataset sizes tested
    fit_times : list of float
        Average fit times for each size
    eval_times : list of float
        Average evaluation times for each size
    """
    fit_times = []
    eval_times = []
    successful_sizes = []

    for n in data_sizes:
        print(f"  Testing n={n}...", end=" ")

        try:
            # Generate test data
            x_train = np.linspace(0, 10, n)
            y_train = np.sin(x_train)
            x_test = np.linspace(0, 10, n * 10)

            fit_time_runs = []
            eval_time_runs = []

            for _ in range(n_runs):
                # Create new instance for each run
                if interpolator_class.__name__ == "HermiteInterpolator":
                    # Hermite needs derivatives
                    derivs = np.cos(x_train)
                    interp = interpolator_class(**kwargs)

                    start = time.perf_counter()
                    interp.fit(x_train, y_train, derivs)
                    fit_time = time.perf_counter() - start
                elif interpolator_class.__name__ == "ChebyshevInterpolator":
                    # Chebyshev only takes y values
                    interp = interpolator_class(n_points=n, x_min=0.0, x_max=10.0, **kwargs)

                    start = time.perf_counter()
                    interp.fit(y_train)
                    fit_time = time.perf_counter() - start
                else:
                    interp = interpolator_class(**kwargs)

                    start = time.perf_counter()
                    interp.fit(x_train, y_train)
                    fit_time = time.perf_counter() - start

                fit_time_runs.append(fit_time)

                # Time evaluation
                start = time.perf_counter()
                _ = interp(x_test)
                eval_time = time.perf_counter() - start
                eval_time_runs.append(eval_time)

            # Average over runs
            fit_times.append(np.mean(fit_time_runs))
            eval_times.append(np.mean(eval_time_runs))
            successful_sizes.append(n)
            print(f"OK (fit: {np.mean(fit_time_runs)*1000:.2f}ms, eval: {np.mean(eval_time_runs)*1000:.2f}ms)")

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    return successful_sizes, fit_times, eval_times


def estimate_complexity(sizes: List[int], times: List[float]) -> str:
    """
    Estimate the algorithmic complexity from timing data.

    Parameters
    ----------
    sizes : list of int
        Dataset sizes
    times : list of float
        Execution times

    Returns
    -------
    complexity : str
        Estimated complexity (e.g., "O(n)", "O(n²)")
    """
    if len(sizes) < 2 or len(times) < 2:
        return "Unknown"

    sizes_arr = np.array(sizes, dtype=float)
    times_arr = np.array(times, dtype=float)

    # Try to fit different complexity models
    # Model: time = a * n^k
    # Taking log: log(time) = log(a) + k*log(n)

    log_sizes = np.log(sizes_arr)
    log_times = np.log(times_arr)

    # Linear regression to find k
    A = np.vstack([log_sizes, np.ones(len(log_sizes))]).T
    k, _ = np.linalg.lstsq(A, log_times, rcond=None)[0]

    # Classify complexity based on exponent k
    if k < 0.5:
        return "O(log n) or O(1)"
    elif 0.5 <= k < 1.2:
        return f"O(n) [k~{k:.2f}]"
    elif 1.2 <= k < 1.8:
        return f"O(n log n) [k~{k:.2f}]"
    elif 1.8 <= k < 2.3:
        return f"O(n^2) [k~{k:.2f}]"
    elif 2.3 <= k < 3.3:
        return f"O(n^3) [k~{k:.2f}]"
    else:
        return f"O(n^{k:.1f})"


def print_scaling_results(
    method_name: str,
    sizes: List[int],
    fit_times: List[float],
    eval_times: List[float]
):
    """Print formatted scaling analysis results."""
    print(f"\n{'='*80}")
    print(f"SCALING ANALYSIS: {method_name}")
    print(f"{'='*80}\n")

    print(f"{'Dataset Size':<15} {'Fit Time (ms)':<20} {'Eval Time (ms)':<20}")
    print(f"{'-'*80}")

    for n, fit_t, eval_t in zip(sizes, fit_times, eval_times):
        print(f"{n:<15} {fit_t*1000:<20.4f} {eval_t*1000:<20.4f}")

    print(f"{'-'*80}\n")

    # Estimate complexity
    fit_complexity = estimate_complexity(sizes, fit_times)
    eval_complexity = estimate_complexity(sizes, eval_times)

    print(f"Estimated Fit Complexity:   {fit_complexity}")
    print(f"Estimated Eval Complexity:  {eval_complexity}")


def main():
    """Run scaling analysis for all interpolation methods."""
    print("\n" + "="*80)
    print("DIMENSIONAL SCALING ANALYSIS FOR INTERLIB")
    print("="*80)

    # Define test sizes (logarithmic spacing)
    small_sizes = [5, 10, 20, 50]
    medium_sizes = [5, 10, 20, 50, 100]
    large_sizes = [5, 10, 20, 50, 100, 200, 500]

    methods = [
        ("Linear", LinearInterpolator, large_sizes, {}),
        ("Quadratic", QuadraticInterpolator, large_sizes, {}),
        ("CubicSpline", CubicSplineInterpolator, large_sizes, {}),
        ("Newton", NewtonInterpolator, medium_sizes, {}),
        ("Lagrange", LagrangeInterpolator, medium_sizes, {}),
        ("Hermite", HermiteInterpolator, medium_sizes, {}),
        ("LeastSquares (deg=3)", LeastSquaresInterpolator, large_sizes, {"degree": 3}),
        ("RBF (Gaussian, e=1.0)", RBFInterpolator, small_sizes, {"kernel": "gaussian", "epsilon": 1.0}),
        ("Chebyshev", ChebyshevInterpolator, medium_sizes, {"use_clenshaw": True}),
    ]

    for method_name, interpolator_class, sizes, kwargs in methods:
        print(f"\nAnalyzing {method_name}...")

        try:
            sizes_tested, fit_times, eval_times = measure_scaling(
                interpolator_class, sizes, n_runs=3, **kwargs
            )

            if len(sizes_tested) >= 2:
                print_scaling_results(method_name, sizes_tested, fit_times, eval_times)
            else:
                print(f"  Insufficient data points for {method_name}")

        except Exception as e:
            print(f"  Failed to analyze {method_name}: {e}")

    print("\n" + "="*80)
    print("SCALING ANALYSIS COMPLETE")
    print("="*80)
    print("\nKEY INSIGHTS:")
    print("- O(n) or O(n log n): Good scalability for large datasets")
    print("- O(n^2): Acceptable for medium datasets (<1000 points)")
    print("- O(n^3): Limited to small datasets (<100 points)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
