import time
from scipy.interpolate import make_interp_spline
from interlib import QuadraticInterpolator

# Common test data: points from y = x²
xs = [0.0, 1.0, 2.0, 3.0, 4.0]
ys = [0.0, 1.0, 4.0, 9.0, 16.0]

eval_points_single = [0.5, 1.5, 2.5, 3.5]
eval_points_array = [0.5, 1.5, 2.5, 3.5]

def benchmark_interlib():
    start_time = time.perf_counter()
    
    # Create and fit interpolator
    quad = QuadraticInterpolator()
    quad.fit(xs, ys)
    
    # Single evaluations
    print("interlib - Single evaluations:")
    for x in eval_points_single:
        true_y = x ** 2
        print(f"x = {x}, y = {quad(x)} (true: {true_y})")
    
    # Multiple evaluations
    results = quad(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    
    # Representation
    print(f"\n{quad}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    
    # Create piecewise quadratic interpolating spline (k=2)
    spl = make_interp_spline(xs, ys, k=2)
    
    print("SciPy - Single evaluations:")
    for x in eval_points_single:
        true_y = x ** 2
        y = spl(x)
        print(f"x = {x}, y = {y} (true: {true_y})")
    
    # Multiple evaluations (vectorized)
    results = spl(eval_points_array)
    print(f"\nSciPy - Multiple evaluations: {results}")
    
    # Representation
    print(f"\n{spl}")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"SciPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

# Run the benchmarks
if __name__ == "__main__":
    print("=== interlib QuadraticInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy make_interp_spline (k=2) ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")