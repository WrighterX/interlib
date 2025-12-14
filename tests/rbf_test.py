from interlib import RBFInterpolator
import time

start_time = time.perf_counter()

# Create RBF interpolator with Gaussian kernel
rbf = RBFInterpolator(kernel="gaussian", epsilon=1.0)

# Known data points (y = sin(x))
xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
ys = [0.0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794]

# Fit
rbf.fit(xs, ys)

# Evaluate at single points
print("Single evaluations (y = sin(x)):")
print(f"x = 0.5, y = {rbf(0.5):.4f} (true: 0.4794)")
print(f"x = 1.5, y = {rbf(1.5):.4f} (true: 0.9975)")
print(f"x = 2.5, y = {rbf(2.5):.4f} (true: 0.5985)")

# Evaluate at multiple points
eval_points = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
results = rbf(eval_points)
print(f"\nMultiple evaluations: {results}")

# Show representation
print(f"\n{rbf}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")