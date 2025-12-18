import time
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from interlib import LagrangeInterpolator

# Common test data
xs = np.array([1.0, 2.0, 3.0])
ys = np.array([1.0, 4.0, 9.0])

eval_points_single = [4.0]
eval_points_array = np.array([1.5, 2.5, 3.5, 4.0])

# Additional exact polynomial test
xs_poly = np.array([0.0, 2.0, 4.0])
ys_poly = np.array([0.0, 4.0, 16.0])  # Exact x^2

def benchmark_interlib():
    start_time = time.perf_counter()
    
    interp = LagrangeInterpolator()
    interp.fit(xs, ys)
    
    print("interlib - Evaluation at x=4.0:", interp(4.0))
    print("interlib - Multiple evaluations:", interp(eval_points_array))
    
    interp_poly = LagrangeInterpolator()
    interp_poly.fit(xs_poly, ys_poly)
    print("\ninterlib - Exact x² test:")
    for x in range(5):
        print(f"P({x}) = {interp_poly(float(x))}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    interp = BarycentricInterpolator(xs, ys)
    
    print("SciPy - Evaluation at x=4.0:", interp(4.0))
    print("SciPy - Multiple evaluations:", interp(eval_points_array))
    
    interp_poly = BarycentricInterpolator(xs_poly, ys_poly)
    print("\nSciPy - Exact x² test:")
    for x in range(5):
        print(f"P({x}) = {interp_poly(float(x))}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=== interlib LagrangeInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy BarycentricInterpolator ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")