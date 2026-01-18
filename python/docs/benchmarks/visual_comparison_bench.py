"""
Visual Comparison Benchmark: interlib vs scipy.interpolate

Generates side-by-side comparison plots showing performance and accuracy
differences between interlib and scipy implementations.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import scipy methods
from scipy.interpolate import (
    BarycentricInterpolator,
    CubicSpline as ScipyCubicSpline,
    RBFInterpolator as ScipyRBFInterpolator,
    interp1d,
    PchipInterpolator
)

# Import interlib methods
from interlib import (
    LagrangeInterpolator,
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    HermiteInterpolator,
    RBFInterpolator
)


def measure_performance_comparison(
    interlib_class,
    scipy_func,
    data_sizes,
    n_runs=3,
    **kwargs
):
    """
    Measure and compare performance between interlib and scipy.

    Returns
    -------
    sizes : list
        Dataset sizes successfully tested
    interlib_times : list
        Total times for interlib (fit + eval) in seconds
    scipy_times : list
        Total times for scipy (fit + eval) in seconds
    """
    sizes_ok = []
    interlib_times = []
    scipy_times = []

    for n in data_sizes:
        try:
            x_train = np.linspace(0, 10, n)
            y_train = np.sin(x_train)
            x_test = np.linspace(0, 10, n * 10)

            # Measure interlib
            il_runs = []
            for _ in range(n_runs):
                if interlib_class.__name__ == "HermiteInterpolator":
                    derivs = np.cos(x_train)
                    interp = interlib_class(**kwargs.get('interlib', {}))
                    start = time.perf_counter()
                    interp.fit(x_train, y_train, derivs)
                    _ = interp(x_test)
                    il_runs.append(time.perf_counter() - start)
                else:
                    interp = interlib_class(**kwargs.get('interlib', {}))
                    start = time.perf_counter()
                    interp.fit(x_train, y_train)
                    _ = interp(x_test)
                    il_runs.append(time.perf_counter() - start)

            # Measure scipy
            sp_runs = []
            for _ in range(n_runs):
                start = time.perf_counter()
                scipy_interp = scipy_func(x_train, y_train, **kwargs.get('scipy', {}))
                _ = scipy_interp(x_test)
                sp_runs.append(time.perf_counter() - start)

            sizes_ok.append(n)
            interlib_times.append(np.mean(il_runs))
            scipy_times.append(np.mean(sp_runs))

        except Exception as e:
            print(f"  Failed at n={n}: {e}")
            continue

    return sizes_ok, interlib_times, scipy_times


def plot_performance_comparison_chart(output_dir="python/benchmark_plots/comparison"):
    """
    Generate performance comparison charts (bar charts and line plots).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nGenerating performance comparison charts...")

    # Define method pairs
    method_pairs = [
        ("Lagrange", LagrangeInterpolator, lambda x, y: BarycentricInterpolator(x, y),
         [10, 20, 50, 100], {}, "Lagrange vs Barycentric"),
        ("Linear", LinearInterpolator, lambda x, y: interp1d(x, y, kind='linear'),
         [50, 100, 200, 500, 1000], {}, "Linear Interpolation"),
        ("CubicSpline", CubicSplineInterpolator, lambda x, y: ScipyCubicSpline(x, y),
         [50, 100, 200, 500, 1000], {}, "Cubic Spline"),
        ("RBF", RBFInterpolator,
         lambda x, y: ScipyRBFInterpolator(x.reshape(-1, 1), y, kernel="gaussian", epsilon=1.0),
         [10, 20, 30], {'interlib': {'kernel': 'gaussian', 'epsilon': 1.0}}, "RBF Gaussian"),
    ]

    for method_name, interlib_class, scipy_func, data_sizes, kwargs, display_name in method_pairs:
        print(f"  Processing {display_name}...")

        sizes, il_times, sp_times = measure_performance_comparison(
            interlib_class, scipy_func, data_sizes, n_runs=3, **kwargs
        )

        if len(sizes) < 2:
            print(f"    Skipped - insufficient data points")
            continue

        # Convert to milliseconds
        il_times_ms = [t * 1000 for t in il_times]
        sp_times_ms = [t * 1000 for t in sp_times]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Bar chart comparison
        x_pos = np.arange(len(sizes))
        width = 0.35

        bars1 = ax1.bar(x_pos - width/2, il_times_ms, width, label='interlib',
                       color='#2E86AB', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, sp_times_ms, width, label='scipy',
                       color='#A23B72', alpha=0.8)

        ax1.set_xlabel('Dataset Size (n)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Total Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{display_name}: Performance Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(s) for s in sizes])
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # Right plot: Speedup ratio
        speedups = [sp/il if il > 0 else 0 for sp, il in zip(sp_times_ms, il_times_ms)]
        colors = ['green' if s > 1 else 'red' for s in speedups]

        ax2.bar(x_pos, speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal Performance')
        ax2.set_xlabel('Dataset Size (n)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Speedup Ratio (scipy/interlib)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{display_name}: Speedup Ratio\n(>1 = interlib faster)',
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([str(s) for s in sizes])
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (pos, speedup) in enumerate(zip(x_pos, speedups)):
            if speedup > 0:
                ax2.text(pos, speedup, f'{speedup:.2f}x',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        safe_name = method_name.lower().replace(' ', '_')
        output_path = Path(output_dir) / f"comparison_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved to: {output_path}")
        plt.close()


def plot_scaling_comparison(output_dir="python/benchmark_plots/comparison"):
    """
    Generate scaling comparison plots (log-log) for multiple methods.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nGenerating scaling comparison plot...")

    methods = [
        ("Linear (interlib)", LinearInterpolator, lambda x, y: interp1d(x, y, kind='linear'),
         [20, 50, 100, 200, 500, 1000], {}, 'interlib'),
        ("Linear (scipy)", LinearInterpolator, lambda x, y: interp1d(x, y, kind='linear'),
         [20, 50, 100, 200, 500, 1000], {}, 'scipy'),
        ("CubicSpline (interlib)", CubicSplineInterpolator, lambda x, y: ScipyCubicSpline(x, y),
         [20, 50, 100, 200, 500, 1000], {}, 'interlib'),
        ("CubicSpline (scipy)", CubicSplineInterpolator, lambda x, y: ScipyCubicSpline(x, y),
         [20, 50, 100, 200, 500, 1000], {}, 'scipy'),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'interlib': ['#2E86AB', '#1B4965'], 'scipy': ['#A23B72', '#F18F01']}
    markers = {'interlib': 'o', 'scipy': 's'}
    linestyles = {'interlib': '-', 'scipy': '--'}

    color_idx = 0
    for label, interlib_class, scipy_func, data_sizes, kwargs, impl in methods:
        print(f"  Processing {label}...")

        sizes, il_times, sp_times = measure_performance_comparison(
            interlib_class, scipy_func, data_sizes, n_runs=3, **kwargs
        )

        if len(sizes) >= 2:
            times_ms = [t * 1000 for t in (il_times if 'interlib' in label else sp_times)]

            color = colors[impl][color_idx // 2]
            ax.loglog(sizes, times_ms, marker=markers[impl], linestyle=linestyles[impl],
                     linewidth=2, markersize=8, label=label, color=color, alpha=0.8)

        color_idx += 1

    ax.set_xlabel('Dataset Size (n)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Scaling Comparison: interlib vs scipy (Log-Log Scale)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "scaling_comparison_all.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def plot_accuracy_comparison(output_dir="python/benchmark_plots/comparison"):
    """
    Generate accuracy comparison plots showing interpolation quality.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nGenerating accuracy comparison plots...")

    # Test function: sin(x) with moderate sampling
    x_train = np.linspace(0, 2*np.pi, 8)
    y_train = np.sin(x_train)
    x_test = np.linspace(0, 2*np.pi, 200)
    y_true = np.sin(x_test)

    method_pairs = [
        ("Linear", LinearInterpolator(), interp1d(x_train, y_train, kind='linear')),
        ("CubicSpline", CubicSplineInterpolator(), ScipyCubicSpline(x_train, y_train)),
        ("Lagrange", LagrangeInterpolator(), BarycentricInterpolator(x_train, y_train)),
    ]

    for method_name, interlib_interp, scipy_interp in method_pairs:
        print(f"  Processing {method_name}...")

        try:
            # Fit interlib
            if method_name == "Linear":
                interlib_interp.fit(x_train, y_train)
            elif method_name == "CubicSpline":
                interlib_interp.fit(x_train, y_train)
            elif method_name == "Lagrange":
                interlib_interp.fit(x_train, y_train)

            # Evaluate both
            y_interlib = interlib_interp(x_test)
            y_scipy = scipy_interp(x_test)

            # Calculate errors
            error_interlib = y_interlib - y_true
            error_scipy = y_scipy - y_true
            rmse_interlib = np.sqrt(np.mean(error_interlib**2))
            rmse_scipy = np.sqrt(np.mean(error_scipy**2))

            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

            # Top plot: Interpolation comparison
            ax1.plot(x_test, y_true, 'k-', linewidth=2, label='True Function', alpha=0.7)
            ax1.plot(x_train, y_train, 'ko', markersize=8, label='Data Points', zorder=5)
            ax1.plot(x_test, y_interlib, '--', linewidth=2, label='interlib', color='#2E86AB')
            ax1.plot(x_test, y_scipy, '-.', linewidth=2, label='scipy', color='#A23B72')
            ax1.set_ylabel('y', fontsize=11, fontweight='bold')
            ax1.set_title(f'{method_name}: Interpolation Comparison', fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Middle plot: Error comparison
            ax2.plot(x_test, error_interlib, linewidth=2, label=f'interlib (RMSE: {rmse_interlib:.6f})',
                    color='#2E86AB')
            ax2.plot(x_test, error_scipy, linewidth=2, label=f'scipy (RMSE: {rmse_scipy:.6f})',
                    color='#A23B72')
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_ylabel('Error', fontsize=11, fontweight='bold')
            ax2.set_title('Interpolation Error Comparison', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)

            # Bottom plot: Absolute error comparison
            abs_error_interlib = np.abs(error_interlib)
            abs_error_scipy = np.abs(error_scipy)
            ax3.semilogy(x_test, abs_error_interlib, linewidth=2, label='interlib',
                        color='#2E86AB')
            ax3.semilogy(x_test, abs_error_scipy, linewidth=2, label='scipy',
                        color='#A23B72')
            ax3.set_xlabel('x', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Absolute Error (log scale)', fontsize=11, fontweight='bold')
            ax3.set_title('Absolute Error Comparison (Log Scale)', fontsize=12, fontweight='bold')
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3, which="both")

            plt.tight_layout()
            safe_name = method_name.lower().replace(' ', '_')
            output_path = Path(output_dir) / f"accuracy_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved to: {output_path}")
            plt.close()

        except Exception as e:
            print(f"    Failed: {e}")


def plot_summary_comparison(output_dir="python/benchmark_plots/comparison"):
    """
    Generate a summary comparison showing all methods at a fixed dataset size.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nGenerating summary comparison...")

    n_points = 100
    x_train = np.linspace(0, 10, n_points)
    y_train = np.sin(x_train)
    x_test = np.linspace(0, 10, n_points * 10)

    methods = [
        ("Linear", LinearInterpolator(), interp1d(x_train, y_train, kind='linear')),
        ("Quadratic", QuadraticInterpolator(), interp1d(x_train, y_train, kind='quadratic')),
        ("CubicSpline", CubicSplineInterpolator(), ScipyCubicSpline(x_train, y_train)),
        ("Lagrange", LagrangeInterpolator(), BarycentricInterpolator(x_train, y_train)),
    ]

    interlib_times = []
    scipy_times = []
    method_names = []

    for method_name, interlib_class, scipy_interp in methods:
        print(f"  Processing {method_name}...")
        try:
            # Measure interlib
            interlib_class.fit(x_train, y_train)
            start = time.perf_counter()
            _ = interlib_class(x_test)
            il_time = (time.perf_counter() - start) * 1000

            # Measure scipy
            start = time.perf_counter()
            _ = scipy_interp(x_test)
            sp_time = (time.perf_counter() - start) * 1000

            method_names.append(method_name)
            interlib_times.append(il_time)
            scipy_times.append(sp_time)

        except Exception as e:
            print(f"    Failed: {e}")

    if len(method_names) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))

        x_pos = np.arange(len(method_names))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, interlib_times, width, label='interlib',
                      color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, scipy_times, width, label='scipy',
                      color='#A23B72', alpha=0.8)

        ax.set_xlabel('Interpolation Method', fontsize=13, fontweight='bold')
        ax.set_ylabel('Evaluation Time (ms)', fontsize=13, fontweight='bold')
        ax.set_title(f'Performance Summary: interlib vs scipy\n(Dataset size: {n_points} points)',
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_path = Path(output_dir) / "summary_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.close()


def main():
    """Run all comparison visualizations."""
    print("="*80)
    print("VISUAL COMPARISON BENCHMARK: interlib vs scipy.interpolate")
    print("="*80)

    output_dir = "python/benchmark_plots/comparison"
    print(f"\nOutput directory: {output_dir}/")

    # Generate all comparison plots
    plot_performance_comparison_chart(output_dir)
    plot_scaling_comparison(output_dir)
    plot_accuracy_comparison(output_dir)
    plot_summary_comparison(output_dir)

    print("\n" + "="*80)
    print("VISUAL COMPARISON BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nAll comparison plots saved to: {output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
