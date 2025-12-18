import time
import math
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from interlib import ChebyshevInterpolator

x_min, x_max = 0.0, 2 * math.pi
n_points = 10

# Chebyshev-Lobatto nodes on [-1, 1], mapped to [x_min, x_max]
t = np.cos(np.linspace(0, np.pi, n_points))
nodes = x_min + (x_max - x_min) * (t + 1) / 2
y_values = np.sin(nodes)

test_points = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

def benchmark_interlib():
    start_time = time.perf_counter()
    
    cheb = ChebyshevInterpolator(n_points=n_points, x_min=x_min, x_max=x_max, use_clenshaw=True)
    cheb.fit(y_values)
    
    print("interlib Chebyshev (Clenshaw) evaluations:")
    for x in test_points:
        val = cheb(x)
        true = math.sin(x)
        err = abs(val - true)
        print(f"x = {x:.1f}, y = {val:.8f} (true: {true:.8f}), error = {err:.2e}")
    
    print(f"\n{cheb}")
    
    elapsed = time.perf_counter() - start_time
    print(f"interlib elapsed time: {elapsed:.6f} seconds")
    return elapsed

def benchmark_numpy():
    start_time = time.perf_counter()
    
    coeffs = np.polynomial.chebyshev.chebfit(t, y_values, deg=n_points-1)
    cheb_poly = Chebyshev(coeffs, domain=[x_min, x_max])
    
    print("NumPy Chebyshev evaluations:")
    for x in test_points:
        val = cheb_poly(x)
        true = math.sin(x)
        err = abs(val - true)
        print(f"x = {x:.1f}, y = {val:.8f} (true: {true:.8f}), error = {err:.2e}")
    
    results = cheb_poly(test_points)
    print(f"\nNumPy multiple evaluations: {results}")
    print(f"\n{cheb_poly}")
    
    elapsed = time.perf_counter() - start_time
    print(f"NumPy elapsed time: {elapsed:.6f} seconds")
    return elapsed

if __name__ == "__main__":
    print("="*60)
    print("CHEBYSHEV INTERPOLATION BENCHMARK")
    print("="*60)
    
    t1 = benchmark_interlib()
    print("\n" + "="*50 + "\n")
    t2 = benchmark_numpy()
    print("\n" + "="*50)
    
    # Final accuracy comparison
    cheb_int = ChebyshevInterpolator(n_points=n_points, x_min=x_min, x_max=x_max)
    cheb_int.fit(y_values)
    diff = np.max(np.abs(cheb_int(test_points) - Chebyshev(np.polynomial.chebyshev.chebfit(t, y_values, n_points-1), [x_min, x_max])(test_points)))
    print(f"Max difference between interlib and NumPy: {diff:.2e}")
    
    print(f"\nSummary:")
    print(f"interlib time : {t1:.6f} s")
    print(f"NumPy time    : {t2:.6f} s")