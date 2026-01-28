"""
Visual Comparison Benchmark: interlib vs scipy.interpolate

Generates side-by-side comparison plots showing performance and accuracy
differences between interlib and scipy implementations.
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import (
    BarycentricInterpolator,
    CubicSpline as ScipyCubicSpline,
    RBFInterpolator as ScipyRBFInterpolator,
    interp1d,
    PchipInterpolator
)#scipy has no chebyshev, least squared ambigiuous alternative

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

# Define method configurations
METHOD_CONFIGS = {
    'lagrange': {
        'interlib_class': LagrangeInterpolator,
        'scipy_func': lambda x, y: BarycentricInterpolator(x, y),
        'default_sizes': [10, 20, 50, 100],
        'kwargs': {'interlib': {}, 'scipy': {}},
        'display_name': 'Lagrange vs Barycentric',
        'needs_derivs': False
    },
    'newton': {
        'interlib_class': NewtonInterpolator,
        'scipy_func': lambda x, y: BarycentricInterpolator(x, y),
        'default_sizes': [10, 20, 50, 100],
        'kwargs': {'interlib': {}, 'scipy': {}},
        'display_name': 'Newton vs Barycentric',
        'needs_derivs': False
    },
    'linear': {
        'interlib_class': LinearInterpolator,
        'scipy_func': lambda x, y: interp1d(x, y, kind='linear'),
        'default_sizes': [50, 100, 200, 500, 1000],
        'kwargs': {'interlib': {}, 'scipy': {}},
        'display_name': 'Linear Interpolation',
        'needs_derivs': False
    },
    'quadratic': {
        'interlib_class': QuadraticInterpolator,
        'scipy_func': lambda x, y: interp1d(x, y, kind='quadratic'),
        'default_sizes': [50, 100, 200, 500, 1000],
        'kwargs': {'interlib': {}, 'scipy': {}},
        'display_name': 'Quadratic Interpolation',
        'needs_derivs': False
    },
    'cubicspline': {
        'interlib_class': CubicSplineInterpolator,
        'scipy_func': lambda x, y: ScipyCubicSpline(x, y),
        'default_sizes': [50, 100, 200, 500, 1000],
        'kwargs': {'interlib': {}, 'scipy': {}},
        'display_name': 'Cubic Spline',
        'needs_derivs': False
    },
    'hermite': {
        'interlib_class': HermiteInterpolator,
        'scipy_func': lambda x, y: PchipInterpolator(x, y),
        'default_sizes': [50, 100, 200, 500, 1000],
        'kwargs': {'interlib': {}, 'scipy': {}},
        'display_name': 'Hermite vs PCHIP',
        'needs_derivs': True
    },
    'rbf': {
        'interlib_class': RBFInterpolator,
        'scipy_func': lambda x, y: ScipyRBFInterpolator(x.reshape(-1, 1), y, kernel="gaussian", epsilon=1.0),
        'default_sizes': [10, 20, 30],
        'kwargs': {'interlib': {'kernel': 'gaussian', 'epsilon': 1.0}, 'scipy': {'kernel': 'gaussian', 'epsilon': 1.0}},
        'display_name': 'RBF Gaussian',
        'needs_derivs': False
    }
}

def measure_performance(
    interlib_class,
    scipy_func,
    data_sizes,
    n_runs=3,
    needs_derivs=False,
    no_compare=False,
    **kwargs
):
    sizes_ok = []
    interlib_times = []
    scipy_times = [] if not no_compare else None

    for n in data_sizes:
        try:
            x_train = np.linspace(0, 10, n)
            y_train = np.sin(x_train)
            x_test = np.linspace(0, 10, n * 10)
            derivs = np.cos(x_train) if needs_derivs else None

            # Measure interlib
            il_runs = []
            for _ in range(n_runs):
                interp = interlib_class(**kwargs.get('interlib', {}))
                start = time.perf_counter()
                if needs_derivs:
                    interp.fit(x_train, y_train, derivs)
                else:
                    interp.fit(x_train, y_train)
                _ = interp(x_test)
                il_runs.append(time.perf_counter() - start)
            interlib_times.append(np.mean(il_runs))

            if not no_compare:
                # Measure scipy
                sp_runs = []
                for _ in range(n_runs):
                    start = time.perf_counter()
                    scipy_interp = scipy_func(x_train, y_train, **kwargs.get('scipy', {}))
                    _ = scipy_interp(x_test)
                    sp_runs.append(time.perf_counter() - start)
                scipy_times.append(np.mean(sp_runs))

            sizes_ok.append(n)

        except Exception as e:
            print(f"  Failed at n={n}: {e}")
            continue

    return sizes_ok, interlib_times, scipy_times

def plot_performance_chart(
    method_key,
    config,
    sizes,
    il_times,
    sp_times,
    output_dir
):
    display_name = config['display_name']
    print(f"  Processing {display_name}...")

    if len(sizes) < 2:
        print(f"    Skipped - insufficient data points")
        return

    # Convert to milliseconds
    il_times_ms = [t * 1000 for t in il_times]
    sp_times_ms = [t * 1000 for t in sp_times] if sp_times else None

    fig, axs = plt.subplots(1, 2 if sp_times else 1, figsize=(14 if sp_times else 7, 6))

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    ax1 = axs[0]
    x_pos = np.arange(len(sizes))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2 if sp_times else x_pos, il_times_ms, width if sp_times else 0.7, label='interlib',
                    color='#2E86AB', alpha=0.8)
    if sp_times:
        bars2 = ax1.bar(x_pos + width/2, sp_times_ms, width, label='scipy',
                        color='#A23B72', alpha=0.8)

    ax1.set_xlabel('Dataset Size (n)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{display_name}: Performance', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    if sp_times:
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    if sp_times:
        ax2 = axs[1]
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

        for i, (pos, speedup) in enumerate(zip(x_pos, speedups)):
            if speedup > 0:
                ax2.text(pos, speedup, f'{speedup:.2f}x',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    safe_name = method_key.lower().replace(' ', '_')
    output_path = Path(output_dir) / f"performance_{safe_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved to: {output_path}")
    plt.close()

def plot_scaling_chart(
    selected_methods,
    args,
    output_dir
):
    print("\nGenerating scaling comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'interlib': ['#2E86AB', '#1B4965', '#0A2342'], 'scipy': ['#A23B72', '#F18F01', '#D36135']}
    markers = {'interlib': 'o', 'scipy': 's'}
    linestyles = {'interlib': '-', 'scipy': '--'}
    color_idx = 0

    for method_key in selected_methods:
        config = METHOD_CONFIGS[method_key]
        sizes = args.sizes or config['default_sizes']
        print(f"  Processing {config['display_name']}...")

        s_ok, il_t, sp_t = measure_performance(
            config['interlib_class'],
            config['scipy_func'],
            sizes,
            n_runs=args.runs,
            needs_derivs=config['needs_derivs'],
            no_compare=args.no_compare,
            **config['kwargs']
        )

        if len(s_ok) >= 2:
            # Plot interlib
            times_ms = [t * 1000 for t in il_t]
            label = f"{method_key.capitalize()} (interlib)"
            ax.loglog(s_ok, times_ms, marker=markers['interlib'], linestyle=linestyles['interlib'],
                      linewidth=2, markersize=8, label=label, color=colors['interlib'][color_idx % len(colors['interlib'])], alpha=0.8)

            if sp_t:
                # Plot scipy
                times_ms = [t * 1000 for t in sp_t]
                label = f"{method_key.capitalize()} (scipy)"
                ax.loglog(s_ok, times_ms, marker=markers['scipy'], linestyle=linestyles['scipy'],
                          linewidth=2, markersize=8, label=label, color=colors['scipy'][color_idx % len(colors['scipy'])], alpha=0.8)

        color_idx += 1

    ax.set_xlabel('Dataset Size (n)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Scaling Comparison: interlib vs scipy (Log-Log Scale)',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "scaling_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

def plot_accuracy_chart(
    method_key,
    config,
    output_dir
):
    display_name = config['display_name']
    print(f"  Processing {display_name} for accuracy...")

    # Test function: sin(x) with moderate sampling
    x_train = np.linspace(0, 2*np.pi, 8)
    y_train = np.sin(x_train)
    x_test = np.linspace(0, 2*np.pi, 200)
    y_true = np.sin(x_test)
    derivs = np.cos(x_train) if config['needs_derivs'] else None

    try:
        # Fit interlib
        interlib_interp = config['interlib_class'](**config['kwargs']['interlib'])
        if config['needs_derivs']:
            interlib_interp.fit(x_train, y_train, derivs)
        else:
            interlib_interp.fit(x_train, y_train)

        y_interlib = interlib_interp(x_test)

        if not args.no_compare:
            scipy_interp = config['scipy_func'](x_train, y_train, **config['kwargs']['scipy'])
            y_scipy = scipy_interp(x_test)
        else:
            y_scipy = None

        # Calculate errors
        error_interlib = y_interlib - y_true
        rmse_interlib = np.sqrt(np.mean(error_interlib**2))
        if y_scipy is not None:
            error_scipy = y_scipy - y_true
            rmse_scipy = np.sqrt(np.mean(error_scipy**2))
        else:
            rmse_scipy = None

        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        # Top plot: Interpolation comparison
        axs[0].plot(x_test, y_true, 'k-', linewidth=2, label='True Function', alpha=0.7)
        axs[0].plot(x_train, y_train, 'ko', markersize=8, label='Data Points', zorder=5)
        axs[0].plot(x_test, y_interlib, '--', linewidth=2, label='interlib', color='#2E86AB')
        if y_scipy is not None:
            axs[0].plot(x_test, y_scipy, '-.', linewidth=2, label='scipy', color='#A23B72')
        axs[0].set_ylabel('y', fontsize=11, fontweight='bold')
        axs[0].set_title(f'{display_name}: Interpolation Comparison', fontsize=13, fontweight='bold')
        axs[0].legend(loc='best', fontsize=10)
        axs[0].grid(True, alpha=0.3)

        # Middle plot: Error comparison
        axs[1].plot(x_test, error_interlib, linewidth=2, label=f'interlib (RMSE: {rmse_interlib:.6f})',
                 color='#2E86AB')
        if y_scipy is not None:
            axs[1].plot(x_test, error_scipy, linewidth=2, label=f'scipy (RMSE: {rmse_scipy:.6f})',
                     color='#A23B72')
        axs[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        axs[1].set_ylabel('Error', fontsize=11, fontweight='bold')
        axs[1].set_title('Interpolation Error Comparison', fontsize=12, fontweight='bold')
        axs[1].legend(loc='best', fontsize=10)
        axs[1].grid(True, alpha=0.3)

        # Bottom plot: Absolute error comparison
        abs_error_interlib = np.abs(error_interlib)
        axs[2].semilogy(x_test, abs_error_interlib, linewidth=2, label='interlib',
                     color='#2E86AB')
        if y_scipy is not None:
            abs_error_scipy = np.abs(error_scipy)
            axs[2].semilogy(x_test, abs_error_scipy, linewidth=2, label='scipy',
                         color='#A23B72')
        axs[2].set_xlabel('x', fontsize=11, fontweight='bold')
        axs[2].set_ylabel('Absolute Error (log scale)', fontsize=11, fontweight='bold')
        axs[2].set_title('Absolute Error Comparison (Log Scale)', fontsize=12, fontweight='bold')
        axs[2].legend(loc='best', fontsize=10)
        axs[2].grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        safe_name = method_key.lower().replace(' ', '_')
        output_path = Path(output_dir) / f"accuracy_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"    Failed: {e}")

def plot_summary_chart(
    selected_methods,
    args,
    output_dir
):
    print("\nGenerating summary comparison...")

    n_points = 100
    x_train = np.linspace(0, 10, n_points)
    y_train = np.sin(x_train)
    x_test = np.linspace(0, 10, n_points * 10)

    interlib_times = []
    scipy_times = [] if not args.no_compare else None
    method_names = []

    for method_key in selected_methods:
        config = METHOD_CONFIGS[method_key]
        print(f"  Processing {config['display_name']}...")
        try:
            derivs = np.cos(x_train) if config['needs_derivs'] else None

            # Measure interlib
            interlib_inst = config['interlib_class'](**config['kwargs']['interlib'])
            if config['needs_derivs']:
                interlib_inst.fit(x_train, y_train, derivs)
            else:
                interlib_inst.fit(x_train, y_train)
            start = time.perf_counter()
            _ = interlib_inst(x_test)
            il_time = (time.perf_counter() - start) * 1000

            if not args.no_compare:
                # Measure scipy
                start = time.perf_counter()
                scipy_interp = config['scipy_func'](x_train, y_train, **config['kwargs']['scipy'])
                _ = scipy_interp(x_test)
                sp_time = (time.perf_counter() - start) * 1000
            else:
                sp_time = None

            method_names.append(method_key.capitalize())
            interlib_times.append(il_time)
            if sp_time is not None:
                scipy_times.append(sp_time)

        except Exception as e:
            print(f"    Failed: {e}")

    if len(method_names) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))

        x_pos = np.arange(len(method_names))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2 if scipy_times else x_pos, interlib_times, width if scipy_times else 0.7, label='interlib',
                       color='#2E86AB', alpha=0.8)
        if scipy_times:
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
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        if scipy_times:
            for bar in bars2:
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
    parser = argparse.ArgumentParser(description="Visual Comparison Benchmark: interlib vs scipy.interpolate")
    parser.add_argument('--methods', nargs='*', default=['all'], help='Methods to benchmark (lagrange, linear, quadratic, cubicspline, hermite, rbf) or all')
    parser.add_argument('--sizes', nargs='*', type=int, help='Dataset sizes (overrides defaults)')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per measurement')
    parser.add_argument('--output-dir', default='python/benches/benchmark_plots', help='Output directory for plots')
    parser.add_argument('--no-compare', action='store_true', help='Benchmark only interlib (no scipy comparison)')
    parser.add_argument('--performance', action='store_true', help='Generate performance charts (per method)')
    parser.add_argument('--scaling', action='store_true', help='Generate scaling comparison plot')
    parser.add_argument('--accuracy', action='store_true', help='Generate accuracy comparison plots (per method)')
    parser.add_argument('--summary', action='store_true', help='Generate summary performance chart')
    args = parser.parse_args()

    if 'all' in args.methods:
        selected_methods = list(METHOD_CONFIGS.keys())
    else:
        selected_methods = [m.lower() for m in args.methods if m.lower() in METHOD_CONFIGS]

    if not selected_methods:
        print("No valid methods selected. Exiting.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_all = not (args.performance or args.scaling or args.accuracy or args.summary)

    print("="*80)
    print("VISUAL COMPARISON BENCHMARK: interlib vs scipy.interpolate")
    print("="*80)
    print(f"Selected methods: {', '.join(selected_methods)}")
    print(f"Output directory: {args.output_dir}")

    if args.performance or run_all:
        print("\nGenerating performance charts...")
        for method_key in selected_methods:
            config = METHOD_CONFIGS[method_key]
            sizes = args.sizes or config['default_sizes']
            s_ok, il_t, sp_t = measure_performance(
                config['interlib_class'],
                config['scipy_func'],
                sizes,
                n_runs=args.runs,
                needs_derivs=config['needs_derivs'],
                no_compare=args.no_compare,
                **config['kwargs']
            )
            plot_performance_chart(method_key, config, s_ok, il_t, sp_t, args.output_dir)

    if args.scaling or run_all:
        plot_scaling_chart(selected_methods, args, args.output_dir)

    if args.accuracy or run_all:
        print("\nGenerating accuracy plots...")
        for method_key in selected_methods:
            config = METHOD_CONFIGS[method_key]
            plot_accuracy_chart(method_key, config, args.output_dir)

    if args.summary or run_all:
        plot_summary_chart(selected_methods, args, args.output_dir)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()