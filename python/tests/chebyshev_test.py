from interlib import ChebyshevInterpolator
import time
import math

start_time = time.perf_counter()

# Test both evaluation methods
print("=" * 60)
print("CHEBYSHEV INTERPOLATION - METHOD COMPARISON")
print("=" * 60)

# Data
x_min, x_max = 0.0, 2*math.pi
test_points = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

# Method 1: Clenshaw algorithm (default, more stable)
print("\n--- CLENSHAW ALGORITHM (DEFAULT) ---")
cheb_clenshaw = ChebyshevInterpolator(n_points=10, x_min=x_min, x_max=x_max, use_clenshaw=True)
nodes = cheb_clenshaw.get_nodes()
y_values = [math.sin(x) for x in nodes]
cheb_clenshaw.fit(y_values)

for x in test_points:
    y_interp = cheb_clenshaw(x)
    y_true = math.sin(x)
    error = abs(y_interp - y_true)
    print(f"x = {x:.1f}, y = {y_interp:.4f} (true: {y_true:.4f}), error = {error:.6f}")

print(f"\n{cheb_clenshaw}")

# Method 2: Direct polynomial evaluation
print("\n--- DIRECT POLYNOMIAL EVALUATION ---")
cheb_direct = ChebyshevInterpolator(n_points=10, x_min=x_min, x_max=x_max, use_clenshaw=False)
cheb_direct.fit(y_values)

for x in test_points:
    y_interp = cheb_direct(x)
    y_true = math.sin(x)
    error = abs(y_interp - y_true)
    print(f"x = {x:.1f}, y = {y_interp:.4f} (true: {y_true:.4f}), error = {error:.6f}")

print(f"\n{cheb_direct}")

# Show they give the same results
results_clenshaw = cheb_clenshaw(test_points)
results_direct = cheb_direct(test_points)
max_diff = max(abs(c - d) for c, d in zip(results_clenshaw, results_direct))
print(f"\nMax difference between methods: {max_diff:.10f}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nElapsed time: {elapsed_time:.6f} seconds")