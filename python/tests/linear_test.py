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
print(linear)

# --- Test update_y ---
print("\n--- update_y tests ---")
linear2 = LinearInterpolator()
linear2.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
val_before = linear2(0.5)
print(f"Before update_y: interp(0.5) = {val_before}")
assert abs(val_before - 0.5) < 1e-10, f"Expected 0.5, got {val_before}"

linear2.update_y([0.0, 2.0, 6.0])
val_after = linear2(0.5)
print(f"After update_y([0.0, 2.0, 6.0]): interp(0.5) = {val_after}")
assert abs(val_after - 1.0) < 1e-10, f"Expected 1.0, got {val_after}"

# Verify endpoints still exact
assert abs(linear2(0.0) - 0.0) < 1e-10
assert abs(linear2(1.0) - 2.0) < 1e-10
assert abs(linear2(2.0) - 6.0) < 1e-10
print("update_y: all assertions passed")

# update_y error: wrong length
try:
    linear2.update_y([1.0, 2.0])
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"update_y wrong length error (expected): {e}")

# update_y error: not fitted
try:
    unfitted = LinearInterpolator()
    unfitted.update_y([1.0, 2.0])
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"update_y not fitted error (expected): {e}")

# --- Test add_point ---
print("\n--- add_point tests ---")
linear3 = LinearInterpolator()
linear3.fit([0.0, 2.0, 4.0], [0.0, 4.0, 8.0])

# Add point in the middle
linear3.add_point(1.0, 1.0)
val = linear3(0.5)
print(f"After add_point(1.0, 1.0): interp(0.5) = {val}")
assert abs(val - 0.5) < 1e-10, f"Expected 0.5, got {val}"

# Verify the new point is exact
assert abs(linear3(1.0) - 1.0) < 1e-10
# Verify old points still exact
assert abs(linear3(0.0) - 0.0) < 1e-10
assert abs(linear3(2.0) - 4.0) < 1e-10
assert abs(linear3(4.0) - 8.0) < 1e-10
# Check interpolation between new point and next old point
val_1_5 = linear3(1.5)
expected_1_5 = 1.0 + (4.0 - 1.0) / (2.0 - 1.0) * 0.5  # 2.5
print(f"interp(1.5) = {val_1_5}, expected {expected_1_5}")
assert abs(val_1_5 - expected_1_5) < 1e-10
print("add_point middle: all assertions passed")

# Add point at the beginning
linear4 = LinearInterpolator()
linear4.fit([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
linear4.add_point(0.0, 0.0)
assert abs(linear4(0.0) - 0.0) < 1e-10
assert abs(linear4(0.5) - 5.0) < 1e-10
assert abs(linear4(1.0) - 10.0) < 1e-10
assert abs(linear4(2.0) - 20.0) < 1e-10
print("add_point beginning: all assertions passed")

# Add point at the end
linear5 = LinearInterpolator()
linear5.fit([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
linear5.add_point(3.0, 6.0)
assert abs(linear5(2.5) - 4.0) < 1e-10
assert abs(linear5(3.0) - 6.0) < 1e-10
assert abs(linear5(1.0) - 1.0) < 1e-10
print("add_point end: all assertions passed")

# add_point error: duplicate x
try:
    linear5.add_point(1.0, 5.0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"add_point duplicate x error (expected): {e}")

# add_point error: not fitted
try:
    unfitted2 = LinearInterpolator()
    unfitted2.add_point(1.0, 2.0)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"add_point not fitted error (expected): {e}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"\nElapsed time: {elapsed_time:.4f} seconds")
print("\nAll linear tests passed!")
