"""
Case Study 2: Signal Reconstruction from Sampled Data
======================================================
This study demonstrates interpolation for signal processing applications:
1. Clean signal reconstruction
2. Noisy signal reconstruction
3. Undersampled signal reconstruction
"""
import time
import numpy as np
from interlib import (
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    RBFInterpolator,
    LeastSquaresInterpolator,
    HermiteInterpolator
)

def generate_test_signal(t, signal_type="composite"):
    """Generate test signals"""
    if signal_type == "composite":
        # Composite signal: sum of sinusoids
        return (np.sin(2 * np.pi * 1.0 * t) +
                0.5 * np.sin(2 * np.pi * 3.0 * t) +
                0.3 * np.sin(2 * np.pi * 7.0 * t))
    elif signal_type == "chirp":
        # Chirp signal (increasing frequency)
        return np.sin(2 * np.pi * (1.0 + 2.0 * t) * t)
    elif signal_type == "square_like":
        # Square-wave-like signal
        return np.sign(np.sin(2 * np.pi * 2.0 * t))

def test_clean_signal_reconstruction():
    """Test interpolation on clean sampled signal"""
    print("=" * 70)
    print("TEST 1: Clean Signal Reconstruction")
    print("=" * 70)
    print("Signal: Composite of three sinusoids")
    print("Sampling: 20 samples over 2 seconds")
   
    # Original high-resolution signal
    t_original = np.linspace(0, 2, 1000)
    signal_original = generate_test_signal(t_original, "composite")
   
    # Sampled signal
    n_samples = 20
    t_sampled = np.linspace(0, 2, n_samples)
    signal_sampled = generate_test_signal(t_sampled, "composite")
   
    # Test reconstruction
    t_test = t_original
   
    results = {}
   
    # 1. Linear Interpolation
    print("\n1. Linear Interpolation")
    start = time.perf_counter()
    linear = LinearInterpolator()
    linear.fit(t_sampled, signal_sampled)
    signal_linear = linear(t_test)
    time_linear = time.perf_counter() - start
    error_linear = np.mean(np.abs(signal_linear - signal_original))
    print(f"   Time: {time_linear:.6f}s, Mean Error: {error_linear:.6e}")
    results['Linear'] = (signal_linear, time_linear, error_linear)
   
    # 2. Quadratic Interpolation
    print("\n2. Quadratic Interpolation")
    start = time.perf_counter()
    quad = QuadraticInterpolator()
    quad.fit(t_sampled, signal_sampled)
    signal_quad = quad(t_test)
    time_quad = time.perf_counter() - start
    error_quad = np.mean(np.abs(signal_quad - signal_original))
    print(f"   Time: {time_quad:.6f}s, Mean Error: {error_quad:.6e}")
    results['Quadratic'] = (signal_quad, time_quad, error_quad)
   
    # 3. Cubic Spline
    print("\n3. Cubic Spline Interpolation")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(t_sampled, signal_sampled)
    signal_spline = spline(t_test)
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(signal_spline - signal_original))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.6e}")
    results['Cubic Spline'] = (signal_spline, time_spline, error_spline)
   
    # 4. RBF (Multiquadric)
    print("\n4. RBF Interpolation (Multiquadric)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="multiquadric", epsilon=2.0)
    rbf.fit(t_sampled, signal_sampled)
    signal_rbf = rbf(t_test)
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(signal_rbf - signal_original))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.6e}")
    results['RBF'] = (signal_rbf, time_rbf, error_rbf)
   
    # Summary
    print("\n" + "-" * 70)
    print(f"{'Method':<20} {'Mean Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (_, t, err) in results.items():
        print(f"{name:<20} {err:>15.6e} {t*1000:>12.3f}")
   
    return results

def test_noisy_signal_reconstruction():
    """Test interpolation on noisy sampled signal"""
    print("\n\n" + "=" * 70)
    print("TEST 2: Noisy Signal Reconstruction")
    print("=" * 70)
    print("Signal: Composite sinusoids + Gaussian noise (SNR ~ 20dB)")
    print("Challenge: Reconstruct underlying signal from noisy samples")
   
    # Original clean signal
    t_original = np.linspace(0, 2, 1000)
    signal_clean = generate_test_signal(t_original, "composite")
   
    # Sampled signal with noise
    np.random.seed(42)
    n_samples = 30
    t_sampled = np.linspace(0, 2, n_samples)
    signal_clean_sampled = generate_test_signal(t_sampled, "composite")
    noise = np.random.normal(0, 0.15, n_samples)
    signal_noisy = signal_clean_sampled + noise
   
    results = {}
   
    # 1. Cubic Spline (passes through all noisy points)
    print("\n1. Cubic Spline (interpolates all points)")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(t_sampled, signal_noisy)
    signal_spline = spline(t_original)
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(signal_spline - signal_clean))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.6e}")
    print(f"   Note: Interpolates noise exactly, may amplify it")
    results['Cubic Spline'] = (signal_spline, time_spline, error_spline)
   
    # 2. Least Squares degree 8 (smooths the noise)
    print("\n2. Least Squares (degree 8, smoothing)")
    start = time.perf_counter()
    ls = LeastSquaresInterpolator(degree=8)
    ls.fit(t_sampled, signal_noisy)
    signal_ls = ls(t_original)
    time_ls = time.perf_counter() - start
    error_ls = np.mean(np.abs(signal_ls - signal_clean))
    r_squared = ls.r_squared()
    print(f"   Time: {time_ls:.6f}s, Mean Error: {error_ls:.6e}")
    print(f"   R² = {r_squared:.6f}")
    print(f"   Note: Does not pass through noisy points, provides smoothing")
    results['Least Squares'] = (signal_ls, time_ls, error_ls)
   
    # 3. RBF with larger epsilon (smoother)
    print("\n3. RBF (Gaussian, epsilon=3.0, smoother)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="gaussian", epsilon=3.0)
    rbf.fit(t_sampled, signal_noisy)
    signal_rbf = rbf(t_original)
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(signal_rbf - signal_clean))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.6e}")
    results['RBF (smooth)'] = (signal_rbf, time_rbf, error_rbf)
   
    # Summary
    print("\n" + "-" * 70)
    print(f"{'Method':<20} {'Mean Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (_, t, err) in results.items():
        print(f"{name:<20} {err:>15.6e} {t*1000:>12.3f}")
   
    print("\nKEY INSIGHT:")
    print("  - Exact interpolation (Cubic Spline) captures noise")
    print("  - Approximation methods (Least Squares) smooth the noise")
    print("  - For noisy data, approximation often beats interpolation!")
   
    return results

def test_undersampled_signal():
    """Test reconstruction of undersampled signal (violates Nyquist)"""
    print("\n\n" + "=" * 70)
    print("TEST 3: Undersampled Signal Reconstruction")
    print("=" * 70)
    print("Signal: Chirp (frequency increases over time)")
    print("Challenge: Nyquist criterion violated - aliasing present")
   
    # Original high-resolution signal
    t_original = np.linspace(0, 2, 2000)
    signal_original = generate_test_signal(t_original, "chirp")
   
    # Severely undersampled
    n_samples = 15
    t_sampled = np.linspace(0, 2, n_samples)
    signal_sampled = generate_test_signal(t_sampled, "chirp")
   
    # Also compute derivatives for Hermite
    # df/dt for chirp: d/dt[sin(2π(1+2t)t)] = 2π(1 + 4t) cos(2π(1+2t)t)
    def chirp_derivative(t):
        return 2 * np.pi * (1 + 4 * t) * np.cos(2 * np.pi * (1 + 2 * t) * t)
   
    derivatives = chirp_derivative(t_sampled)
   
    results = {}
   
    # 1. Linear (baseline)
    print("\n1. Linear Interpolation (baseline)")
    start = time.perf_counter()
    linear = LinearInterpolator()
    linear.fit(t_sampled, signal_sampled)
    signal_linear = linear(t_original)
    time_linear = time.perf_counter() - start
    error_linear = np.mean(np.abs(signal_linear - signal_original))
    print(f"   Time: {time_linear:.6f}s, Mean Error: {error_linear:.6e}")
    results['Linear'] = (signal_linear, time_linear, error_linear)
   
    # 2. Cubic Spline
    print("\n2. Cubic Spline")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(t_sampled, signal_sampled)
    signal_spline = spline(t_original)
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(signal_spline - signal_original))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.6e}")
    results['Cubic Spline'] = (signal_spline, time_spline, error_spline)
   
    # 3. Hermite (uses derivative information)
    print("\n3. Hermite Interpolation (with derivative information)")
    start = time.perf_counter()
    hermite = HermiteInterpolator()
    hermite.fit(t_sampled, signal_sampled, derivatives)
    signal_hermite = hermite(t_original)
    time_hermite = time.perf_counter() - start
    error_hermite = np.mean(np.abs(signal_hermite - signal_original))
    print(f"   Time: {time_hermite:.6f}s, Mean Error: {error_hermite:.6e}")
    print(f"   Note: Uses both signal values AND derivatives")
    results['Hermite'] = (signal_hermite, time_hermite, error_hermite)
   
    # 4. RBF
    print("\n4. RBF (Gaussian, epsilon=2.0)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="gaussian", epsilon=2.0)
    rbf.fit(t_sampled, signal_sampled)
    signal_rbf = rbf(t_original)
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(signal_rbf - signal_original))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.6e}")
    results['RBF'] = (signal_rbf, time_rbf, error_rbf)
   
    # Summary
    print("\n" + "-" * 70)
    print(f"{'Method':<20} {'Mean Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (_, t, err) in results.items():
        print(f"{name:<20} {err:>15.6e} {t*1000:>12.3f}")
   
    print("\nKEY INSIGHT:")
    print("  - Undersampling causes all methods to have significant error")
    print("  - Hermite (with derivatives) performs best - extra information helps!")
    print("  - Smooth interpolation cannot recover aliased high frequencies")
    print("  - Solution: Increase sampling rate (satisfy Nyquist criterion)")
   
    return results

def main():
    """Run all signal reconstruction tests"""
    print("\n" + "#" * 70)
    print("# CASE STUDY 2: SIGNAL RECONSTRUCTION FROM SAMPLED DATA")
    print("#" * 70)
   
    total_start = time.perf_counter()
   
    # Test 1: Clean signal
    clean_results = test_clean_signal_reconstruction()
   
    # Test 2: Noisy signal
    noisy_results = test_noisy_signal_reconstruction()
   
    # Test 3: Undersampled signal
    undersampled_results = test_undersampled_signal()
   
    total_time = time.perf_counter() - total_start
   
    print("\n" + "=" * 70)
    print(f"Total execution time: {total_time:.3f} seconds")
    print("=" * 70)
   
    print("\n\nCONCLUSIONS:")
    print("-" * 70)
    print("1. Clean Signals:")
    print("   - Cubic Spline and RBF provide excellent reconstruction")
    print("   - Linear interpolation adequate for slowly varying signals")
    print("\n2. Noisy Signals:")
    print("   - Exact interpolation propagates noise")
    print("   - Approximation methods (Least Squares) provide denoising")
    print("   - Trade-off: fitting accuracy vs. noise smoothing")
    print("\n3. Undersampled Signals:")
    print("   - All methods fail when Nyquist criterion violated")
    print("   - Hermite interpolation (with derivatives) helps")
    print("   - Proper sampling is more important than interpolation method!")

if __name__ == "__main__":
    main()