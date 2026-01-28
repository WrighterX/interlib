"""
Visual Benchmark Suite for interlib

Generates performance curves and interpolation accuracy plots using Matplotlib.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from interlib import (
    LagrangeInterpolator,
    NewtonInterpolator,
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    HermiteInterpolator,
    LeastSquaresInterpolator,
    RBFInterpolator
)


def measure_performance(interpolator_class, data_sizes, n_runs=3, **kwargs):
    """
    Measure performance across different dataset sizes.

    Returns
    -------
    sizes : list
        Dataset sizes successfully tested
    fit_times : list
        Average fit times in seconds
    eval_times : list
        Average evaluation times in seconds
    """
    sizes_ok = []
    fit_times = []
    eval_times = []

    for n in data_sizes:
        try:
            x_train = np.linspace(0, 10, n)
            y_train = np.sin(x_train)
            x_test = np.linspace(0, 10, n * 10)

            fit_runs = []
            eval_runs = []

            for _ in range(n_runs):
                if interpolator_class.__name__ == "HermiteInterpolator":
                    derivs = np.cos(x_train)
                    interp = interpolator_class(**kwargs)
                    start = time.perf_counter()
                    interp.fit(x_train, y_train, derivs)
                    fit_runs.append(time.perf_counter() - start)
                else:
                    interp = interpolator_class(**kwargs)
                    start = time.perf_counter()
                    interp.fit(x_train, y_train)
                    fit_runs.append(time.perf_counter() - start)

                start = time.perf_counter()
                _ = interp(x_test)
                eval_runs.append(time.perf_counter() - start)

            sizes_ok.append(n)
            fit_times.append(np.mean(fit_runs))
            eval_times.append(np.mean(eval_runs))

        except Exception as e:
            print(f"  Failed at n={n}: {e}")
            continue

    return sizes_ok, fit_times, eval_times


def plot_performance_curves(output_dir="python/benchmark_plots"):
    """
    Generate performance curves (Time vs. Data Size) in log-log scale.
    """
    Path(output_dir).mkdir(exist_ok=True)

    methods = [
        ("Linear", LinearInterpolator, [10, 20, 50, 100, 200, 500, 1000], {}),
        ("Quadratic", QuadraticInterpolator, [10, 20, 50, 100, 200, 500, 1000], {}),
        ("CubicSpline", CubicSplineInterpolator, [10, 20, 50, 100, 200, 500, 1000], {}),
        ("Newton", NewtonInterpolator, [10, 20, 50, 100, 200], {}),
        ("Lagrange", LagrangeInterpolator, [10, 20, 50, 100, 200], {}),
        ("Hermite", HermiteInterpolator, [10, 20, 50, 100, 200], {}),
        ("LeastSquares", LeastSquaresInterpolator, [10, 20, 50, 100, 200, 500], {"degree": 3}),
        ("RBF", RBFInterpolator, [10, 20, 50], {"kernel": "gaussian", "epsilon": 1.0}),
    ]

    print("\nGenerating performance curves...")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method_name, interpolator_class, data_sizes, kwargs in methods:
        print(f"  Processing {method_name}...")

        sizes, fit_times, eval_times = measure_performance(
            interpolator_class, data_sizes, n_runs=3, **kwargs
        )

        if len(sizes) >= 2:
            # Convert to milliseconds
            fit_times_ms = [t * 1000 for t in fit_times]
            eval_times_ms = [t * 1000 for t in eval_times]

            ax1.loglog(sizes, fit_times_ms, 'o-', label=method_name, linewidth=2, markersize=6)
            ax2.loglog(sizes, eval_times_ms, 'o-', label=method_name, linewidth=2, markersize=6)

    # Configure fit time plot
    ax1.set_xlabel('Dataset Size (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fit Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Fit Time vs Dataset Size (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Configure evaluation time plot
    ax2.set_xlabel('Dataset Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Evaluation Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Evaluation Time vs Dataset Size (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "performance_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved performance curves to: {output_path}")
    plt.close()


def plot_interpolation_accuracy(output_dir="python/benchmark_plots"):
    """
    Generate interpolation accuracy plots showing:
    - Original function
    - Interpolated values
    - Residuals (errors)
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Test function: sin(x) on [0, 2*pi]
    x_train = np.linspace(0, 2*np.pi, 10)
    y_train = np.sin(x_train)
    x_test = np.linspace(0, 2*np.pi, 200)
    y_true = np.sin(x_test)

    methods = [
        ("Linear", LinearInterpolator(), None),
        ("Quadratic", QuadraticInterpolator(), None),
        ("CubicSpline", CubicSplineInterpolator(), None),
        ("Lagrange", LagrangeInterpolator(), None),
        ("Hermite", HermiteInterpolator(), np.cos(x_train)),
        ("LeastSquares (deg=5)", LeastSquaresInterpolator(degree=5), None),
    ]

    print("\nGenerating interpolation accuracy plots...")

    for method_name, interpolator, extra_data in methods:
        print(f"  Processing {method_name}...")

        try:
            # Fit the interpolator
            if extra_data is not None:  # Hermite needs derivatives
                interpolator.fit(x_train, y_train, extra_data)
            else:
                interpolator.fit(x_train, y_train)

            # Evaluate
            y_interp = interpolator(x_test)

            # Calculate residuals
            residuals = y_interp - y_true
            max_error = np.max(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Top plot: Original function vs Interpolation
            ax1.plot(x_test, y_true, 'b-', linewidth=2, label='True Function (sin x)', alpha=0.7)
            ax1.plot(x_train, y_train, 'ro', markersize=8, label='Training Points', zorder=5)
            ax1.plot(x_test, y_interp, 'g--', linewidth=2, label=f'{method_name} Interpolation')
            ax1.set_xlabel('x', fontsize=11, fontweight='bold')
            ax1.set_ylabel('y', fontsize=11, fontweight='bold')
            ax1.set_title(f'{method_name}: Interpolation vs True Function', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Bottom plot: Residuals
            ax2.plot(x_test, residuals, 'r-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax2.fill_between(x_test, residuals, alpha=0.3, color='red')
            ax2.set_xlabel('x', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Residual Error', fontsize=11, fontweight='bold')
            ax2.set_title(f'Residuals (Max Error: {max_error:.6f}, RMSE: {rmse:.6f})',
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            output_path = Path(output_dir) / f"accuracy_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {output_path}")
            plt.close()

        except Exception as e:
            print(f"    Failed: {e}")


def plot_method_comparison(output_dir="python/benchmark_plots"):
    """
    Generate a comparison plot showing all methods on the same function.
    """
    Path(output_dir).mkdir(exist_ok=True)

    print("\nGenerating method comparison plot...")

    # Test function with varying curvature
    x_train = np.array([0, 1, 2, 3, 4, 5, 6])
    y_train = np.sin(x_train) + 0.3 * x_train
    x_test = np.linspace(0, 6, 200)
    y_true = np.sin(x_test) + 0.3 * x_test

    methods = [
        ("Linear", LinearInterpolator(), None),
        ("Quadratic", QuadraticInterpolator(), None),
        ("CubicSpline", CubicSplineInterpolator(), None),
        ("Lagrange", LagrangeInterpolator(), None),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot true function
    ax.plot(x_test, y_true, 'b-', linewidth=3, label='True Function', alpha=0.7)
    ax.plot(x_train, y_train, 'ko', markersize=10, label='Training Points', zorder=10)

    # Plot each method
    colors = ['red', 'green', 'orange', 'purple']
    linestyles = ['--', '-.', ':', '--']

    for (method_name, interpolator, extra_data), color, ls in zip(methods, colors, linestyles):
        try:
            if extra_data is not None:
                interpolator.fit(x_train, y_train, extra_data)
            else:
                interpolator.fit(x_train, y_train)

            y_interp = interpolator(x_test)
            ax.plot(x_test, y_interp, linestyle=ls, color=color, linewidth=2,
                   label=method_name, alpha=0.8)

        except Exception as e:
            print(f"  Failed for {method_name}: {e}")

    ax.set_xlabel('x', fontsize=13, fontweight='bold')
    ax.set_ylabel('y', fontsize=13, fontweight='bold')
    ax.set_title('Interpolation Method Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "method_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved method comparison to: {output_path}")
    plt.close()


def main():
    """Run all visual benchmark generations."""
    print("="*80)
    print("VISUAL BENCHMARK SUITE FOR INTERLIB")
    print("="*80)

    output_dir = "benchmark_plots"
    print(f"\nOutput directory: {output_dir}/")

    # Generate all plots
    plot_performance_curves(output_dir)
    plot_interpolation_accuracy(output_dir)
    plot_method_comparison(output_dir)

    print("\n" + "="*80)
    print("VISUAL BENCHMARK GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
