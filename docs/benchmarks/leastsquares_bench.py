import time
import numpy as np
from interlib import LeastSquaresInterpolator

# Common test data: points from y = 2 + 3x - 0.5x² (exact quadratic)
xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ys = [2.0, 3.125, 3.5, 3.125, 2.0, 0.125, -2.5, -5.875, -10.0]

eval_points_single = [0.25, 1.75, 3.25]
eval_points_array = np.array([0.25, 1.75, 3.25])

def true_function(x):
    return 2.0 + 3.0 * x - 0.5 * x**2

def benchmark_interlib():
    start_time = time.perf_counter()
    
    # Create interpolator with degree 2
    ls = LeastSquaresInterpolator(degree=2)
    ls.fit(xs, ys)
    
    # Coefficients and R-squared
    coeffs = ls.get_coefficients()
    r_sq = ls.r_squared()
    print("interlib - Least Squares Approximation (degree 2):")
    print(f"Coefficients: {coeffs}")
    print(f"R-squared: {r_sq:.6f}")
    
    # Single evaluations
    print("\ninterlib - Single evaluations:")
    for x in eval_points_single:
        true_y = true_function(x)
        print(f"x = {x}, y = {ls(x):.4f} (true: {true_y:.4f})")
    
    # Multiple evaluations
    results = ls(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    
    # Representation
    print(f"\n{ls}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy_numpy():
    start_time = time.perf_counter()
    
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    
    # Fit polynomial of degree 2
    coeffs = np.polyfit(xs_arr, ys_arr, 2)  # coeffs[0] is highest degree
    
    # Create callable polynomial
    poly = np.poly1d(coeffs)
    
    # Compute R-squared
    y_pred = poly(xs_arr)
    ss_res = np.sum((ys_arr - y_pred) ** 2)
    ss_tot = np.sum((ys_arr - np.mean(ys_arr)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    
    print("SciPy/NumPy - Least Squares Approximation (degree 2):")
    print(f"Coefficients (highest to lowest degree): {coeffs.tolist()}")
    print(f"R-squared: {r_sq:.6f}")
    
    # Single evaluations
    print("\nSciPy/NumPy - Single evaluations:")
    for x in eval_points_single:
        true_y = true_function(x)
        y = poly(x)
        print(f"x = {x}, y = {y:.4f} (true: {true_y:.4f})")
    
    # Multiple evaluations (vectorized)
    results = poly(eval_points_array)
    print(f"\nSciPy/NumPy - Multiple evaluations: {results}")
    
    # Representation
    print(f"\n{poly}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"SciPy/NumPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

# Run the benchmarks
if __name__ == "__main__":
    print("=== interlib LeastSquaresInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy/NumPy np.polyfit + np.poly1d ===\n")
    time_scipy = benchmark_scipy_numpy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy/NumPy time : {time_scipy:.6f} s")