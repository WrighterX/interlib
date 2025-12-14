import time
import math
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from interlib import ChebyshevInterpolator

# Domain and function
x_min, x_max = 0.0, 2 * math.pi
n_points = 10  # Degree + 1 points (n_points=10 means degree 9 polynomial)

nodes = np.cos(np.linspace(0, np.pi, n_points))[::-1]  # Reversed to increasing order
mapped_nodes = x_min + (x_max - x_min) * (nodes + 1) / 2
y_values = np.sin(mapped_nodes)

test_points_single = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
test_points_array = np.array(test_points_single)

def benchmark_interlib():
    start_time = time.perf_counter()
    
    print("--- interlib ChebyshevInterpolator (Clenshaw - default) ---")
    cheb = ChebyshevInterpolator(n_points=n_points, x_min=x_min, x_max=x_max, use_clenshaw=True)
    cheb.fit(y_values)
    
    for x in test_points_single:
        y_interp = cheb(x)
        y_true = math.sin(x)
        error = abs(y_interp - y_true)
        print(f"x = {x:.1f}, y = {y_interp:.4f} (true: {y_true:.4f}), error = {error:.6f}")
    
    print(f"\n{cheb}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_numpy():
    start_time = time.perf_counter()
    
    print("--- NumPy Chebyshev (on mapped Lobatto nodes) ---")
    
    # Fit Chebyshev series on [-1, 1] first
    coeffs = np.polynomial.chebyshev.chebfit(nodes, y_values, deg=n_points-1)
    
    # Create callable on original domain
    cheb_poly = Chebyshev(coeffs, domain=[x_min, x_max])
    
    for x in test_points_single:
        y_interp = cheb_poly(x)
        y_true = math.sin(x)
        error = abs(y_interp - y_true)
        print(f"x = {x:.1f}, y = {y_interp:.4f} (true: {y_true:.4f}), error = {error:.6f}")
    
    # Multiple evaluations
    results = cheb_poly(test_points_array)
    print(f"\nNumPy multiple evaluations: {results}")
    
    # Representation
    print(f"\n{cheb_poly}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"NumPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

# Run the benchmarks
if __name__ == "__main__":
    print("=" * 60)
    print("CHEBYSHEV INTERPOLATION BENCHMARK")
    print("=" * 60)
    
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    time_numpy = benchmark_numpy()
    print("\n" + "="*40)
    
    # Compare multiple evaluation results
    cheb_clenshaw = ChebyshevInterpolator(n_points=n_points, x_min=x_min, x_max=x_max)
    cheb_clenshaw.fit(y_values)
    results_interlib = cheb_clenshaw(test_points_array)
    
    coeffs = np.polynomial.chebyshev.chebfit(nodes, y_values, deg=n_points-1)
    cheb_poly = Chebyshev(coeffs, domain=[x_min, x_max])
    results_numpy = cheb_poly(test_points_array)
    
    max_diff = np.max(np.abs(results_interlib - results_numpy))
    print(f"Max difference between interlib (Clenshaw) and NumPy Chebyshev: {max_diff:.10f}")
    
    print(f"\nSummary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"NumPy time    : {time_numpy:.6f} s")