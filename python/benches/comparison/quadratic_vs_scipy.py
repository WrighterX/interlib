"""
Quadratic Interpolator: interlib vs SciPy

IMPORTANT: scipy has NO direct equivalent to interlib's QuadraticInterpolator.

interlib QuadraticInterpolator:
  - Algorithm: Local overlapping triplets — each interval uses 3 nearest points
  - Continuity: C⁰ only (values match, derivatives may be discontinuous)
  - Scope: Local (each evaluation depends on only 3 data points)

scipy interp1d(kind='quadratic'):
  - Algorithm: B-spline of order 2 via FITPACK (confirmed: equivalent to make_interp_spline(k=2))
  - Continuity: C¹ (values AND first derivatives are continuous)
  - Scope: Global (all data points influence the result)

These are fundamentally different algorithms that produce different results on
non-polynomial data. This benchmark is therefore NOT a fair apples-to-apples
comparison — it exists only to show relative performance characteristics.

For a true comparison, interlib's method is closest to manually looping over
triplets, which scipy does not provide as a built-in.
"""

import time
import numpy as np
from scipy.interpolate import interp1d
from interlib import QuadraticInterpolator


def benchmark(label, fn, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    avg = np.mean(times)
    print(f"  {label:<35} {avg:.4f} ms")
    return avg


def compare_at_size(n_data, n_eval, n_runs=5):
    print(f"\n{'='*65}")
    print(f"Dataset: {n_data} points, Eval: {n_eval} points")
    print(f"{'='*65}")

    x_data = np.linspace(0, 10, n_data)
    y_data = np.sin(x_data)
    x_test = np.linspace(0, 10, n_eval)

    # --- SciPy ---
    print("\nSciPy interp1d(kind='quadratic'):")
    scipy_fit = benchmark("fit", lambda: interp1d(x_data, y_data, kind='quadratic'), n_runs)
    scipy_interp = interp1d(x_data, y_data, kind='quadratic')
    scipy_eval = benchmark("eval (NumPy array)", lambda: scipy_interp(x_test), n_runs)
    scipy_total = scipy_fit + scipy_eval

    # --- interlib (list input) ---
    print("\ninterlib QuadraticInterpolator (list input):")
    x_list = x_data.tolist()
    y_list = y_data.tolist()
    x_test_list = x_test.tolist()

    interlib_fit = benchmark("fit", lambda: QuadraticInterpolator().fit(x_list, y_list), n_runs)

    interp = QuadraticInterpolator()
    interp.fit(x_list, y_list)
    interlib_eval_list = benchmark("eval (list)", lambda: interp(x_test_list), n_runs)

    # --- interlib (NumPy array input) ---
    print("\ninterlib QuadraticInterpolator (NumPy array input):")
    interlib_eval_numpy = benchmark("eval (NumPy array)", lambda: interp(x_test), n_runs)

    interlib_total_numpy = interlib_fit + interlib_eval_numpy

    # --- Summary ---
    print(f"\n{'Summary':}")
    print(f"  {'Metric':<30} {'SciPy':>10} {'interlib':>10} {'Ratio':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Fit time (ms)':<30} {scipy_fit:>10.4f} {interlib_fit:>10.4f} {scipy_fit/interlib_fit:>9.2f}x")
    print(f"  {'Eval time NumPy (ms)':<30} {scipy_eval:>10.4f} {interlib_eval_numpy:>10.4f} {scipy_eval/interlib_eval_numpy:>9.2f}x")
    print(f"  {'Total time (ms)':<30} {scipy_total:>10.4f} {interlib_total_numpy:>10.4f} {scipy_total/interlib_total_numpy:>9.2f}x")

    return {
        'n_data': n_data, 'n_eval': n_eval,
        'scipy_fit': scipy_fit, 'scipy_eval': scipy_eval,
        'interlib_fit': interlib_fit,
        'interlib_eval_list': interlib_eval_list,
        'interlib_eval_numpy': interlib_eval_numpy,
    }


def main():
    print("="*65)
    print("QUADRATIC INTERPOLATOR — interlib vs SciPy")
    print("="*65)

    results = []
    for n_data in [50, 100, 500, 1000, 5000]:
        n_eval = n_data * 10
        r = compare_at_size(n_data, n_eval)
        results.append(r)

    print(f"\n\n{'='*65}")
    print("SCALING SUMMARY")
    print(f"{'='*65}")
    print(f"\n{'n_data':<8} {'SciPy eval':>12} {'interlib(np)':>14} {'Speedup':>10}")
    print(f"{'-'*50}")
    for r in results:
        speedup = r['scipy_eval'] / r['interlib_eval_numpy']
        winner = "interlib faster" if speedup > 1 else "scipy faster"
        print(f"{r['n_data']:<8} {r['scipy_eval']:>12.4f} {r['interlib_eval_numpy']:>14.4f} {speedup:>9.2f}x  ({winner})")

    print(f"\n{'='*65}")
    print("NOTE: interlib uses Rust + pre-computed coefficients + binary search")
    print("      SciPy uses optimized C/Fortran via NumPy")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
