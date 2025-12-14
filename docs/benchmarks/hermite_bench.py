import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from interlib import HermiteInterpolator

# Common test data: points from y = x³ with derivatives dy/dx = 3x²
xs = [0.0, 1.0, 2.0, 3.0]
ys = [0.0, 1.0, 8.0, 27.0]
dys = [0.0, 3.0, 12.0, 27.0]

eval_points_single = [0.5, 1.5, 2.5]
eval_points_array = np.array([0.5, 1.5, 2.5])

def benchmark_interlib():
    start_time = time.perf_counter()
    
    # Create and fit interpolator
    hermite = HermiteInterpolator()
    hermite.fit(xs, ys, dys)
    
    # Single evaluations
    print("interlib - Single evaluations (y = x³):")
    for x in eval_points_single:
        true_y = x ** 3
        print(f"x = {x}, y = {hermite(x):.4f} (true: {true_y:.4f})")
    
    # Multiple evaluations
    results = hermite(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    
    # Representation
    print(f"\n{hermite}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    # Create monotonic cubic Hermite spline (PCHIP-style slopes would differ; use explicit derivatives)
    spl = CubicHermiteSpline(xs, ys, dys)
    
    print("SciPy - Single evaluations (y = x³):")
    for x in eval_points_single:
        true_y = x ** 3
        y = spl(x)
        print(f"x = {x}, y = {y:.4f} (true: {true_y:.4f})")
    
    # Multiple evaluations (vectorized)
    results = spl(eval_points_array)
    print(f"\nSciPy - Multiple evaluations: {results}")
    
    # Representation (shows knots, coeffs, etc.)
    print(f"\n{spl}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

# Run the benchmarks
if __name__ == "__main__":
    print("=== interlib HermiteInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy CubicHermiteSpline ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")