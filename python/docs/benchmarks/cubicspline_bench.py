import time
import numpy as np
from scipy.interpolate import CubicSpline
from interlib import CubicSplineInterpolator

# Common test data: points from y = x²
xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

eval_points_single = [0.5, 1.5, 2.5, 3.5]
eval_points_array = np.array(eval_points_single)

def benchmark_interlib():
    start_time = time.perf_counter()
    
    spline = CubicSplineInterpolator()
    spline.fit(xs, ys)
    
    print("interlib - Single evaluations:")
    for x in eval_points_single:
        true_y = x ** 2
        print(f"x = {x}, y = {spline(x):.6f} (true: {true_y:.6f})")
    
    results = spline(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    
    elapsed = time.perf_counter() - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    # Use CubicSpline with default 'not-a-knot' (matches make_interp_spline k=3 default)
    spl = CubicSpline(xs, ys)  # bc_type='not-a-knot' by default
    
    print("SciPy - Single evaluations:")
    for x in eval_points_single:
        true_y = x ** 2
        y = spl(x)
        print(f"x = {x}, y = {y:.6f} (true: {true_y:.6f})")
    
    results = spl(eval_points_array)
    print(f"\nSciPy - Multiple evaluations: {results}")
    
    elapsed = time.perf_counter() - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=== interlib CubicSplineInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy CubicSpline (not-a-knot) ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")