import time
import numpy as np
from interlib import LeastSquaresInterpolator

# Exact quadratic: y = 2 + 3x - 0.5x²
xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
ys = 2.0 + 3.0*xs - 0.5*xs**2

eval_points_single = [0.25, 1.75, 3.25]
eval_points_array = np.array(eval_points_single)

def true_func(x): return 2.0 + 3.0*x - 0.5*x*x

def benchmark_interlib():
    start_time = time.perf_counter()
    
    ls = LeastSquaresInterpolator(degree=2)
    ls.fit(xs, ys)
    
    coeffs = ls.get_coefficients()
    r2 = ls.r_squared()
    
    print("interlib - Least Squares (degree 2):")
    print(f"Coefficients: {coeffs}")
    print(f"R² = {r2:.10f} (should be 1.0)")
    
    print("\ninterlib - Single evaluations:")
    for x in eval_points_single:
        val = ls(x)
        true = true_func(x)
        print(f"x = {x}, y = {val:.6f} (true: {true:.6f})")
    
    results = ls(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    print(f"\n{ls}")
    
    elapsed = time.perf_counter() - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_numpy():
    start_time = time.perf_counter()
    
    coeffs = np.polyfit(xs, ys, 2)
    poly = np.poly1d(coeffs)
    
    y_pred = poly(xs)
    ss_res = np.sum((ys - y_pred)**2)
    ss_tot = np.sum((ys - np.mean(ys))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 1.0
    
    print("NumPy - Least Squares (degree 2):")
    print(f"Coefficients (high→low): {coeffs.tolist()}")
    print(f"R² = {r2:.10f} (should be 1.0)")
    
    print("\nNumPy - Single evaluations:")
    for x in eval_points_single:
        val = poly(x)
        true = true_func(x)
        print(f"x = {x}, y = {val:.6f} (true: {true:.6f})")
    
    results = poly(eval_points_array)
    print(f"\nNumPy - Multiple evaluations: {results}")
    print(f"\n{poly}")
    
    elapsed = time.perf_counter() - start_time
    print(f"NumPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=== interlib LeastSquaresInterpolator ===\n")
    t1 = benchmark_interlib()
    print("\n" + "="*50 + "\n")
    print("=== NumPy np.polyfit + np.poly1d ===\n")
    t2 = benchmark_numpy()
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"interlib time : {t1:.6f} s")
    print(f"NumPy time    : {t2:.6f} s")