from interlib import HermiteInterpolator
import time

start_time = time.perf_counter()

# Create interpolator
hermite = HermiteInterpolator()

# Known data points for y = x^3
# Provide both function values and derivatives (dy/dx = 3x^2)
xs = [0.0, 1.0, 2.0, 3.0]
ys = [0.0, 1.0, 8.0, 27.0]  # x^3
dys = [0.0, 3.0, 12.0, 27.0]  # 3x^2

# Fit
hermite.fit(xs, ys, dys)

# Evaluate at single points
print("Single evaluations (y = x^3):")
print(f"x = 0.5, y = {hermite(0.5):.4f} (true: 0.1250)")
print(f"x = 1.5, y = {hermite(1.5):.4f} (true: 3.3750)")
print(f"x = 2.5, y = {hermite(2.5):.4f} (true: 15.6250)")

# Evaluate at multiple points
eval_points = [0.5, 1.5, 2.5]
results = hermite(eval_points)
print(f"\nMultiple evaluations: {results}")

# Show representation
print(f"\n{hermite}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")