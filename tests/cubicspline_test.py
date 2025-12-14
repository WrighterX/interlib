from interlib import CubicSplineInterpolator
import time

start_time = time.perf_counter()

# Create interpolator
spline = CubicSplineInterpolator()

# Known data points (y = x^2)
xs = [0.0, 1.0, 2.0, 3.0, 4.0]
ys = [0.0, 1.0, 4.0, 9.0, 16.0]

# Fit
spline.fit(xs, ys)

# Evaluate at single points
print("Single evaluations:")
print(f"x = 0.5, y = {spline(0.5)} (true: 0.25)")
print(f"x = 1.5, y = {spline(1.5)} (true: 2.25)")
print(f"x = 2.5, y = {spline(2.5)} (true: 6.25)")
print(f"x = 3.5, y = {spline(3.5)} (true: 12.25)")

# Evaluate at multiple points
eval_points = [0.5, 1.5, 2.5, 3.5]
results = spline(eval_points)
print(f"\nMultiple evaluations: {results}")

# Show representation
print(f"\n{spline}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")