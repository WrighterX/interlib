import time
from scipy.interpolate import make_interp_spline
from interlib import LinearInterpolator

# Common test data
xs = [0.0, 2.0, 4.0]
ys = [0.0, 4.0, 8.0]
eval_points_single = [1.0, 3.0, 3.5]
eval_points_array = [1.0, 3.0, 3.5]

def benchmark_interlib():
    start_time = time.perf_counter()
    # Create and fit interpolator
    linear = LinearInterpolator()
    linear.fit(xs, ys)
    # Single evaluations
    print("interlib - Single evaluations:")
    for x in eval_points_single:
        print(f"x = {x}, y = {linear(x)}")
    # Multiple evaluations
    results = linear(eval_points_array)
    print(f"\ninterlib - Multiple evaluations: {results}")
    # Representation
    print(f"\n{linear}")
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_scipy():
    start_time = time.perf_counter()
    # Create the linear spline (k=1 ensures piecewise linear interpolation)
    spl = make_interp_spline(xs, ys, k=1)
    
    print("SciPy - Single evaluations:")
    for x in eval_points_single:
        y = spl(x)  # spl behaves like a callable function
        print(f"x = {x}, y = {y}")
    
    # Multiple evaluations (vectorized over list or array)
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
    print("=== interlib LinearInterpolator ===\n")
    time_interlib = benchmark_interlib()
    print("\n" + "="*40 + "\n")
    print("=== SciPy make_interp_spline (linear) ===\n")
    time_scipy = benchmark_scipy()
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"SciPy time    : {time_scipy:.6f} s")