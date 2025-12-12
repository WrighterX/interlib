"""A comparing module that provides Lagrange Interpolation's elapsed times from SciPy and interlib libraries."""

from scipy.interpolate import BarycentricInterpolator
from interlib import LagrangeInterpolator
import time

def scipy_lagrange():
    print("SciPy Lagrange Solution")
    start_time = time.perf_counter()

    # Create interpolator instance and fit data (can be done in one step)
    x = [1.0, 2.0, 3.0]
    y = [1.0, 4.0, 9.0]
    interp = BarycentricInterpolator(x, y)  # fits automatically on creation

    # Evaluate at a single point
    unknown_y = interp(4.0)
    print(f"Interpolated value at x=4: {unknown_y}")

    # Evaluate at multiple points (accepts array-like)
    x_new = [1.5, 2.5, 3.5, 4.0]
    y_new = interp(x_new)
    print(f"Interpolated values: {y_new}")

    x_vals = [0.0, 2.0, 4.0]
    y_vals = [0.0, 4.0, 16.0]  # x^2 polynomial

    interp2 = BarycentricInterpolator(x_vals, y_vals)

    print("\nTesting P(x) = x² interpolation:")
    for x in range(5):
        px = interp2(float(x))
        print(f"P({x}) = {px}")

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"\nElapsed time (SciPy): {elapsed_time:.4f} seconds")

def interlib_lagrange():
    print("interlib Lagrange Solution")
    start_time = time.perf_counter()

    # Create interpolator instance
    interp = LagrangeInterpolator()

    # Fit with data points
    x = [1.0, 2.0, 3.0]
    y = [1.0, 4.0, 9.0]
    interp.fit(x, y)

    # Evaluate at a single point
    unknown_y = interp(4.0)
    print(f"Interpolated value at x=4: {unknown_y}")

    # Evaluate at multiple points
    x_new = [1.5, 2.5, 3.5, 4.0]
    y_new = interp(x_new)
    print(f"Interpolated values: {y_new}")

    x_vals = [0.0, 2.0, 4.0]
    y_vals = [0.0, 4.0, 16.0]

    interp2 = LagrangeInterpolator()
    interp2.fit(x_vals, y_vals)

    for x in range(5):
        px = interp2(float(x))
        print(f"P({x}) = {px}")

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time (interlib): {elapsed_time:.4f} seconds")

scipy_lagrange()
print("\n")
interlib_lagrange()