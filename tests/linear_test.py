from interlib import LinearInterpolator
import time

start_time = time.perf_counter()

# Create interpolator
linear = LinearInterpolator()

# Known data points
xs = [0.0, 2.0, 4.0]
ys = [0.0, 4.0, 8.0]

# Fit
linear.fit(xs, ys)

# Evaluate at single points
print("Single evaluations:")
print(f"x = 1.0, y = {linear(1.0)}")
print(f"x = 3.0, y = {linear(3.0)}")
print(f"x = 3.5, y = {linear(3.5)}")

# Evaluate at multiple points
eval_points = [1.0, 3.0, 3.5]
results = linear(eval_points)
print(f"\nMultiple evaluations: {results}")

# Show representation
print(f"\n{linear}")

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")