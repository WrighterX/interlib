import time
import numpy as np
from scipy.interpolate import RBFInterpolator
from interlib import RBFInterpolator as InterlibRBF

# Common test data: y = sin(x)
xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ys = np.sin(xs)

# SciPy requires 2D input for coordinates
xs_2d = xs.reshape(-1, 1)

eval_single = [0.5, 1.5, 2.5]
eval_array = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
eval_array_2d = eval_array.reshape(-1, 1)

def benchmark_interlib():
    start_time = time.perf_counter()
    
    rbf = InterlibRBF(kernel="gaussian", epsilon=1.0)
    rbf.fit(xs, ys)
    
    print("interlib RBF (Gaussian, ε=1.0) - Single evaluations:")
    for x in eval_single:
        val = rbf(x)
        true = np.sin(x)
        print(f"x = {x}, y = {val:.6f} (true: {true:.6f})")
    
    results = rbf(eval_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    print(f"\n{rbf}")
    
    elapsed = time.perf_counter() - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    # Fixed constructor call - works on SciPy <1.10 and >=1.10
    rbf = RBFInterpolator(xs_2d, ys, kernel="gaussian", epsilon=1.0)
    
    print("SciPy RBF (Gaussian, ε=1.0) - Single evaluations:")
    for x in eval_single:
        val = rbf(np.array([[x]]))[0]
        true = np.sin(x)
        print(f"x = {x}, y = {val:.6f} (true: {true:.6f})")
    
    results = rbf(eval_array_2d).flatten()
    print(f"\nSciPy - Multiple evaluations: {results}")
    print(f"\n{rbf}")
    
    elapsed = time.perf_counter() - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=== interlib RBFInterpolator ===\n")
    t1 = benchmark_interlib()
    print("\n" + "="*50 + "\n")
    print("=== SciPy RBFInterpolator ===\n")
    t2 = benchmark_scipy()
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"interlib time : {t1:.6f} s")
    print(f"SciPy time    : {t2:.6f} s")