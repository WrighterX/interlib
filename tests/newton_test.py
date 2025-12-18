from interlib import NewtonInterpolator
import time

start_time = time.perf_counter()

# Create interpolator
newton = NewtonInterpolator()

# Sample data (y = x²)
xs = [1.0, 2.0, 3.0, 4.0]
ys = [1.0, 4.0, 9.0, 16.0]

# Fit
newton.fit(xs, ys)

# Get coefficients
print("Newton coefficients:")
coef = newton.get_coefficients()
for i, c in enumerate(coef):
    print(f"a{i} = {c}")

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
print(f"Elapsed time: {elapsed_time:.4f} seconds")