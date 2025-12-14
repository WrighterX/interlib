"""A comparing module that provides Newton Interpolation's elapsed times from SciPy and interlib libraries."""

from scipy.interpolate import lagrange
from interlib import NewtonInterpolator
import time

def scipy_newton():
    print("SciPy Newton Solution")
    start_time = time.perf_counter()
    # Sample data (y = x²)
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 4.0, 9.0, 16.0]

    # Fit
    newton = lagrange(xs, ys)

    # Evaluate at points
    print("\nEvaluations:")
    print(f"P(2.5) = {newton(2.5)}")
    print(f"P(5.0) = {newton(5.0)}")

    # Evaluate at multiple points
    eval_points = [2.5, 3.5, 5.0]
    results = newton(eval_points)
    print(f"\nMultiple evaluations: {results}")

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time (SciPy): {elapsed_time:.4f} seconds")

def interlib_newton():
    print("interlib Newton Solution")
    start_time = time.perf_counter()
    # Create interpolator
    newton = NewtonInterpolator()

    # Sample data (y = x²)
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 4.0, 9.0, 16.0]

    # Fit
    newton.fit(xs, ys)

    # Evaluate at points
    print("\nEvaluations:")
    print(f"P(2.5) = {newton(2.5)}")
    print(f"P(5.0) = {newton(5.0)}")

    # Evaluate at multiple points
    eval_points = [2.5, 3.5, 5.0]
    results = newton(eval_points)
    print(f"\nMultiple evaluations: {results}")

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time (interlib): {elapsed_time:.4f} seconds")

scipy_newton()
print("\n")
interlib_newton()