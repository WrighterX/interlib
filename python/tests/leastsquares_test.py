from interlib import LeastSquaresInterpolator
import time

start_time = time.perf_counter()

# Create interpolator with degree 2 (quadratic)
ls = LeastSquaresInterpolator(degree=2)

# Known data points (y = 2 + 3x - 0.5x^2)
xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ys = [2.0, 3.125, 3.5, 3.125, 2.0, 0.125, -2.5, -5.875, -10.0]

# Fit
ls.fit(xs, ys)

# Get coefficients and R-squared
coeffs = ls.get_coefficients()
r_sq = ls.r_squared()

print("Least Squares Approximation (degree 2):")
print(f"Coefficients: {coeffs}")
print(f"R-squared: {r_sq:.6f}")

# Evaluate at single points
print("\nSingle evaluations:")
print(f"x = 0.25, y = {ls(0.25):.4f} (true: 2.7188)")
print(f"x = 1.75, y = {ls(1.75):.4f} (true: 2.7188)")
print(f"x = 3.25, y = {ls(3.25):.4f} (true: -4.2188)")

# Evaluate at multiple points
eval_points = [0.25, 1.75, 3.25]
results = ls(eval_points)
print(f"\nMultiple evaluations: {results}")

# Show representation
print(f"\n{ls}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")