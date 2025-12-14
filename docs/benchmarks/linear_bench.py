import time
import numpy as np
from interlib import LinearInterpolator

# Common test data
xs = [0.0, 2.0, 4.0]
ys = [0.0, 4.0, 8.0]
eval_points_single = [1.0, 3.0, 3.5]
eval_points_array = np.array([1.0, 3.0, 3.5])


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


def benchmark_numpy():
    start_time = time.perf_counter()

    # np.interp works directly without a class instance
    print("NumPy - Single evaluations:")
    for x in eval_points_single:
        y = np.interp(x, xs, ys)
        print(f"x = {x}, y = {y}")

    # Multiple evaluations (vectorized)
    results = np.interp(eval_points_array, xs, ys)
    print(f"\nNumPy - Multiple evaluations: {results}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"NumPy elapsed time: {elapsed:.6f} seconds")
    return elapsed


# Run the benchmarks
if __name__ == "__main__":
    print("=== interlib LinearInterpolator ===\n")
    time_interlib = benchmark_interlib()

    print("\n" + "="*40 + "\n")

    print("=== NumPy np.interp ===\n")
    time_numpy = benchmark_numpy()

    print("\n" + "="*40)
    print(f"Summary:")
    print(f"interlib time : {time_interlib:.6f} s")
    print(f"NumPy time    : {time_numpy:.6f} s")