import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from interlib import HermiteInterpolator

# Test data: y = x³, dy/dx = 3x²
xs = np.array([0.0, 1.0, 2.0, 3.0])
ys = np.array([0.0, 1.0, 8.0, 27.0])
dys = np.array([0.0, 3.0, 12.0, 27.0])

eval_points_single = [0.5, 1.5, 2.5]
eval_points_array = np.array([0.5, 1.5, 2.5])

def benchmark_interlib():
    start_time = time.perf_counter()
    
    hermite = HermiteInterpolator()
    hermite.fit(xs, ys, dys)
    
    print("interlib - Single evaluations (y = x³):")
    for x in eval_points_single:
        true = x**3
        val = hermite(x)
        print(f"x = {x}, y = {val:.6f} (true: {true:.6f}), error = {abs(val - true):.2e}")
    
    results = hermite(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    print(f"\n{hermite}")
    
    elapsed = time.perf_counter() - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    spl = CubicHermiteSpline(xs, ys, dys)
    
    print("SciPy - Single evaluations (y = x³):")
    for x in eval_points_single:
        true = x**3
        val = spl(x)
        print(f"x = {x}, y = {val:.6f} (true: {true:.6f}), error = {abs(val - true):.2e}")
    
    results = spl(eval_points_array)
    print(f"\nSciPy - Multiple evaluations: {results}")
    print(f"\n{spl}")
    
    elapsed = time.perf_counter() - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=== interlib HermiteInterpolator ===\n")
    t1 = benchmark_interlib()
    print("\n" + "="*50 + "\n")
    print("=== SciPy CubicHermiteSpline ===\n")
    t2 = benchmark_scipy()
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"interlib time : {t1:.6f} s")
    print(f"SciPy time    : {t2:.6f} s")