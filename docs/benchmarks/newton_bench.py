import time
import numpy as np
from scipy.interpolate import BarycentricInterpolator  # Newton form via barycentric is equivalent and stable
from interlib import NewtonInterpolator

# Common test data: exact x²
xs = np.array([1.0, 2.0, 3.0, 4.0])
ys = np.array([1.0, 4.0, 9.0, 16.0])

eval_points_single = [2.5, 5.0]
eval_points_array = np.array([2.5, 3.5, 5.0])

def benchmark_interlib():
    start_time = time.perf_counter()
    
    newton = NewtonInterpolator()
    newton.fit(xs, ys)
    
    print("interlib - Single evaluations:")
    for x in eval_points_single:
        print(f"P({x}) = {newton(x)} (true: {x**2})")
    
    results = newton(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    
    elapsed = time.perf_counter() - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    # Barycentric implements stable Newton-form evaluation
    interp = BarycentricInterpolator(xs, ys)
    
    print("SciPy - Single evaluations:")
    for x in eval_points_single:
        print(f"P({x}) = {interp(x)} (true: {x**2})")
    
    results = interp(eval_points_array)
    print(f"\nSciPy - Multiple evaluations: {results}")
    
    elapsed = time.perf_counter() - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=== interlib NewtonInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy BarycentricInterpolator (Newton form) ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")