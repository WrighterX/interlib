import time
import numpy as np
from scipy.interpolate import RBFInterpolator as SciPyRBFInterpolator
from interlib import RBFInterpolator

# Common test data: y = sin(x)
xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ys = np.sin(xs)  # Exact values for reproducibility

# For SciPy, data must be shaped as (n_samples, n_features) i.e. column vector for 1D
xs_scipy = xs.reshape(-1, 1)

eval_points_single = [0.5, 1.5, 2.5]
eval_points_array = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

# For SciPy multiple evaluations
eval_points_array_scipy = eval_points_array.reshape(-1, 1)

def benchmark_interlib():
    start_time = time.perf_counter()
    
    # Create RBF interpolator with Gaussian kernel
    rbf = RBFInterpolator(kernel="gaussian", epsilon=1.0)
    rbf.fit(xs, ys)
    
    print("interlib - Single evaluations (y = sin(x)):")
    for x in eval_points_single:
        true_y = np.sin(x)
        print(f"x = {x}, y = {rbf(x):.4f} (true: {true_y:.4f})")
    
    # Multiple evaluations
    results = rbf(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    
    # Representation
    print(f"\n{rbf}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    # SciPy RBFInterpolator with Gaussian kernel
    rbf = SciPyRBFInterpolator(xs_scipy, ys, kernel='gaussian', epsilon=1.0)
    
    print("SciPy - Single evaluations (y = sin(x)):")
    for x in eval_points_single:
        true_y = np.sin(x)
        # Single point: reshape to (1, 1)
        y = rbf(np.array([[x]]))[0]  # Extract scalar from array
        print(f"x = {x}, y = {y:.4f} (true: {true_y:.4f})")
    
    # Multiple evaluations (vectorized)
    results = rbf(eval_points_array_scipy).flatten()
    print(f"\nSciPy - Multiple evaluations: {results}")
    
    # Representation (limited in SciPy)
    print(f"\n{rbf}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

# Run the benchmarks
if __name__ == "__main__":
    print("=== interlib RBFInterpolator (Gaussian) ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy RBFInterpolator (Gaussian) ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")