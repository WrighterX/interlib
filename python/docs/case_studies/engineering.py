"""
Case Study 3: Engineering Dataset Interpolation
================================================
This study demonstrates interpolation on realistic engineering data:
1. Temperature profile across a heated rod
2. Pressure-Volume relationship (thermodynamics)
3. Material stress-strain curve
"""

import time
import math
import numpy as np
from interlib import (
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    RBFInterpolator,
    LeastSquaresInterpolator,
    HermiteInterpolator
)


def test_temperature_profile():
    """Test interpolation on temperature distribution data"""
    print("=" * 70)
    print("TEST 1: Temperature Profile Across a Heated Rod")
    print("=" * 70)
    print("Scenario: 1-meter aluminum rod, heated at one end")
    print("Data: Temperature measurements at 11 positions")
    
    # Measured positions (meters) and temperatures (°C)
    # Left end heated to 100°C, right end at room temp (20°C)
    # Follows approximate exponential decay with some measurement noise
    positions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Simulated temperature data (exponential decay + small noise)
    np.random.seed(42)
    temp_ideal = 20 + 80 * np.exp(-2.5 * positions)
    noise = np.random.normal(0, 1.5, len(positions))
    temperatures = temp_ideal + noise
    
    # Ensure boundary conditions
    temperatures[0] = 100.0  # Hot end
    temperatures[-1] = 20.0  # Cold end
    
    print(f"\nMeasured data:")
    print(f"{'Position (m)':<15} {'Temperature (°C)':<20}")
    print("-" * 35)
    for pos, temp in zip(positions, temperatures):
        print(f"{pos:<15.2f} {temp:<20.2f}")
    
    # High-resolution positions for interpolation
    pos_fine = np.linspace(0, 1, 200)
    temp_ideal_fine = 20 + 80 * np.exp(-2.5 * pos_fine)
    
    results = {}
    
    # 1. Linear Interpolation
    print("\n1. Linear Interpolation")
    start = time.perf_counter()
    linear = LinearInterpolator()
    linear.fit(positions.tolist(), temperatures.tolist())
    temp_linear = np.array([linear(float(p)) for p in pos_fine])
    time_linear = time.perf_counter() - start
    error_linear = np.mean(np.abs(temp_linear - temp_ideal_fine))
    print(f"   Time: {time_linear:.6f}s, Mean Error: {error_linear:.3f}°C")
    results['Linear'] = (temp_linear, time_linear, error_linear)
    
    # 2. Cubic Spline
    print("\n2. Cubic Spline Interpolation")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(positions.tolist(), temperatures.tolist())
    temp_spline = np.array([spline(float(p)) for p in pos_fine])
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(temp_spline - temp_ideal_fine))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.3f}°C")
    results['Cubic Spline'] = (temp_spline, time_spline, error_spline)
    
    # 3. Least Squares (exponential-like fit)
    print("\n3. Least Squares (degree 5)")
    start = time.perf_counter()
    ls = LeastSquaresInterpolator(degree=5)
    ls.fit(positions.tolist(), temperatures.tolist())
    temp_ls = np.array([ls(float(p)) for p in pos_fine])
    time_ls = time.perf_counter() - start
    error_ls = np.mean(np.abs(temp_ls - temp_ideal_fine))
    r_squared = ls.r_squared()
    print(f"   Time: {time_ls:.6f}s, Mean Error: {error_ls:.3f}°C")
    print(f"   R² = {r_squared:.6f}")
    results['Least Squares'] = (temp_ls, time_ls, error_ls)
    
    # 4. RBF
    print("\n4. RBF (Gaussian, epsilon=5.0)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="gaussian", epsilon=5.0)
    rbf.fit(positions.tolist(), temperatures.tolist())
    temp_rbf = np.array([rbf(float(p)) for p in pos_fine])
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(temp_rbf - temp_ideal_fine))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.3f}°C")
    results['RBF'] = (temp_rbf, time_rbf, error_rbf)
    
    # Summary
    print("\n" + "-" * 70)
    print("Engineering Query: What is the temperature at position 0.45m?")
    print("-" * 70)
    print(f"{'Method':<20} {'Temp at 0.45m':>15} {'Mean Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (temp_curve, t, err) in results.items():
        idx = np.argmin(np.abs(pos_fine - 0.45))
        temp_at_045 = temp_curve[idx]
        print(f"{name:<20} {temp_at_045:>15.2f}°C {err:>15.3f}°C {t*1000:>12.3f}")
    
    ideal_at_045 = 20 + 80 * np.exp(-2.5 * 0.45)
    print("-" * 70)
    print(f"{'Ideal value':<20} {ideal_at_045:>15.2f}°C")
    
    return results


def test_pressure_volume():
    """Test interpolation on P-V relationship (thermodynamics)"""
    print("\n\n" + "=" * 70)
    print("TEST 2: Pressure-Volume Relationship (Isothermal Process)")
    print("=" * 70)
    print("Scenario: Ideal gas isothermal compression/expansion")
    print("Theory: PV = constant (follows hyperbola)")
    
    # Measured data points (P in kPa, V in liters)
    # Following PV = 10 (approximately) with measurement errors
    np.random.seed(123)
    V_measured = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    P_ideal = 10.0 / V_measured  # PV = 10
    noise = np.random.normal(0, 0.2, len(V_measured))
    P_measured = P_ideal + noise
    
    print(f"\nMeasured data:")
    print(f"{'Volume (L)':<15} {'Pressure (kPa)':<20} {'PV Product':<15}")
    print("-" * 50)
    for v, p in zip(V_measured, P_measured):
        print(f"{v:<15.2f} {p:<20.3f} {p*v:<15.3f}")
    
    # Fine grid for interpolation
    V_fine = np.linspace(0.5, 10.0, 200)
    P_ideal_fine = 10.0 / V_fine
    
    results = {}
    
    # 1. Linear (will fail for hyperbolic relationship)
    print("\n1. Linear Interpolation")
    start = time.perf_counter()
    linear = LinearInterpolator()
    linear.fit(V_measured.tolist(), P_measured.tolist())
    P_linear = np.array([linear(float(v)) for v in V_fine])
    time_linear = time.perf_counter() - start
    error_linear = np.mean(np.abs(P_linear - P_ideal_fine))
    print(f"   Time: {time_linear:.6f}s, Mean Error: {error_linear:.3f} kPa")
    print(f"   Note: Poor for non-linear relationships")
    results['Linear'] = (P_linear, time_linear, error_linear)
    
    # 2. Cubic Spline
    print("\n2. Cubic Spline Interpolation")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(V_measured.tolist(), P_measured.tolist())
    P_spline = np.array([spline(float(v)) for v in V_fine])
    time_spline = time.perf_counter() - start
    error_spline = np.mean(np.abs(P_spline - P_ideal_fine))
    print(f"   Time: {time_spline:.6f}s, Mean Error: {error_spline:.3f} kPa")
    results['Cubic Spline'] = (P_spline, time_spline, error_spline)
    
    # 3. RBF (should handle non-linearity well)
    print("\n3. RBF (Multiquadric, epsilon=1.0)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="multiquadric", epsilon=1.0)
    rbf.fit(V_measured.tolist(), P_measured.tolist())
    P_rbf = np.array([rbf(float(v)) for v in V_fine])
    time_rbf = time.perf_counter() - start
    error_rbf = np.mean(np.abs(P_rbf - P_ideal_fine))
    print(f"   Time: {time_rbf:.6f}s, Mean Error: {error_rbf:.3f} kPa")
    results['RBF'] = (P_rbf, time_rbf, error_rbf)
    
    # 4. Least Squares (degree 6 for smooth curve)
    print("\n4. Least Squares (degree 6)")
    start = time.perf_counter()
    ls = LeastSquaresInterpolator(degree=6)
    ls.fit(V_measured.tolist(), P_measured.tolist())
    P_ls = np.array([ls(float(v)) for v in V_fine])
    time_ls = time.perf_counter() - start
    error_ls = np.mean(np.abs(P_ls - P_ideal_fine))
    r_squared = ls.r_squared()
    print(f"   Time: {time_ls:.6f}s, Mean Error: {error_ls:.3f} kPa")
    print(f"   R² = {r_squared:.6f}")
    results['Least Squares'] = (P_ls, time_ls, error_ls)
    
    # Summary
    print("\n" + "-" * 70)
    print("Engineering Query: What is the pressure at volume 2.5L?")
    print("-" * 70)
    print(f"{'Method':<20} {'P at 2.5L':>15} {'Mean Error':>15} {'Time (ms)':>12}")
    print("-" * 70)
    for name, (P_curve, t, err) in results.items():
        idx = np.argmin(np.abs(V_fine - 2.5))
        P_at_25 = P_curve[idx]
        print(f"{name:<20} {P_at_25:>15.3f} kPa {err:>15.3f} kPa {t*1000:>12.3f}")
    
    ideal_at_25 = 10.0 / 2.5
    print("-" * 70)
    print(f"{'Ideal value':<20} {ideal_at_25:>15.3f} kPa")
    
    return results


def test_stress_strain():
    """Test interpolation on material stress-strain curve"""
    print("\n\n" + "=" * 70)
    print("TEST 3: Material Stress-Strain Curve")
    print("=" * 70)
    print("Scenario: Tensile test of steel specimen")
    print("Data: Stress vs. Strain with elastic and plastic regions")
    
    # Stress-strain data for steel (MPa vs. strain)
    # Elastic region (linear), then yield point, then plastic region
    strain = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 
                       0.010, 0.020, 0.040, 0.060, 0.080, 0.100])
    
    # Realistic stress-strain behavior
    stress = np.array([0, 200, 400, 600, 800, 950,    # Elastic region
                       1100, 1200, 1250, 1280, 1300, 1310])  # Plastic region
    
    # Add small measurement noise
    np.random.seed(456)
    noise = np.random.normal(0, 10, len(strain))
    stress_measured = stress + noise
    stress_measured[0] = 0  # Zero stress at zero strain
    
    print(f"\nMeasured data:")
    print(f"{'Strain':<15} {'Stress (MPa)':<20}")
    print("-" * 35)
    for e, s in zip(strain, stress_measured):
        print(f"{e:<15.4f} {s:<20.1f}")
    
    # Fine grid
    strain_fine = np.linspace(0, 0.10, 300)
    
    results = {}
    
    # 1. Linear (piecewise linear)
    print("\n1. Linear Interpolation")
    start = time.perf_counter()
    linear = LinearInterpolator()
    linear.fit(strain.tolist(), stress_measured.tolist())
    stress_linear = np.array([linear(float(e)) for e in strain_fine])
    time_linear = time.perf_counter() - start
    print(f"   Time: {time_linear:.6f}s")
    print(f"   Note: Captures piecewise behavior but not smooth")
    results['Linear'] = (stress_linear, time_linear)
    
    # 2. Quadratic (local parabolic fit)
    print("\n2. Quadratic Interpolation")
    start = time.perf_counter()
    quad = QuadraticInterpolator()
    quad.fit(strain.tolist(), stress_measured.tolist())
    stress_quad = np.array([quad(float(e)) for e in strain_fine])
    time_quad = time.perf_counter() - start
    print(f"   Time: {time_quad:.6f}s")
    results['Quadratic'] = (stress_quad, time_quad)
    
    # 3. Cubic Spline
    print("\n3. Cubic Spline Interpolation")
    start = time.perf_counter()
    spline = CubicSplineInterpolator()
    spline.fit(strain.tolist(), stress_measured.tolist())
    stress_spline = np.array([spline(float(e)) for e in strain_fine])
    time_spline = time.perf_counter() - start
    print(f"   Time: {time_spline:.6f}s")
    print(f"   Note: Smooth curve, good for material models")
    results['Cubic Spline'] = (stress_spline, time_spline)
    
    # 4. Hermite with computed derivatives
    print("\n4. Hermite Interpolation (with derivative estimates)")
    # Estimate derivatives using finite differences
    derivatives = np.zeros(len(strain))
    derivatives[0] = (stress_measured[1] - stress_measured[0]) / (strain[1] - strain[0])
    for i in range(1, len(strain) - 1):
        derivatives[i] = (stress_measured[i+1] - stress_measured[i-1]) / (strain[i+1] - strain[i-1])
    derivatives[-1] = (stress_measured[-1] - stress_measured[-2]) / (strain[-1] - strain[-2])
    
    start = time.perf_counter()
    hermite = HermiteInterpolator()
    hermite.fit(strain.tolist(), stress_measured.tolist(), derivatives.tolist())
    stress_hermite = np.array([hermite(float(e)) for e in strain_fine])
    time_hermite = time.perf_counter() - start
    print(f"   Time: {time_hermite:.6f}s")
    print(f"   Note: Enforces smoothness with derivative matching")
    results['Hermite'] = (stress_hermite, time_hermite)
    
    # 5. RBF
    print("\n5. RBF (Thin Plate Spline)")
    start = time.perf_counter()
    rbf = RBFInterpolator(kernel="thin_plate_spline", epsilon=1.0)
    rbf.fit(strain.tolist(), stress_measured.tolist())
    stress_rbf = np.array([rbf(float(e)) for e in strain_fine])
    time_rbf = time.perf_counter() - start
    print(f"   Time: {time_rbf:.6f}s")
    results['RBF'] = (stress_rbf, time_rbf)
    
    # Summary - compute elastic modulus (slope in elastic region)
    print("\n" + "-" * 70)
    print("Engineering Analysis:")
    print("-" * 70)
    
    # Elastic modulus from strain 0.001 to 0.003
    strain_elastic_range = (0.001, 0.003)
    
    print(f"\n{'Method':<20} {'Elastic Modulus (GPa)':>25} {'Time (ms)':>12}")
    print("-" * 70)
    
    for name, data in results.items():
        stress_curve = data[0]
        t = data[1]
        
        # Find stress at elastic range points
        idx1 = np.argmin(np.abs(strain_fine - strain_elastic_range[0]))
        idx2 = np.argmin(np.abs(strain_fine - strain_elastic_range[1]))
        
        stress1 = stress_curve[idx1]
        stress2 = stress_curve[idx2]
        
        # Elastic modulus = ΔStress / ΔStrain
        E_modulus = (stress2 - stress1) / (strain_elastic_range[1] - strain_elastic_range[0])
        E_modulus_GPa = E_modulus / 1000  # Convert to GPa
        
        print(f"{name:<20} {E_modulus_GPa:>25.1f} {t*1000:>12.3f}")
    
    print("-" * 70)
    print("Expected for steel: ~200 GPa")
    print("\nNote: Elastic modulus is the slope in the linear elastic region")
    
    return results


def main():
    """Run all engineering dataset tests"""
    print("\n" + "#" * 70)
    print("# CASE STUDY 3: ENGINEERING DATASET INTERPOLATION")
    print("#" * 70)
    
    total_start = time.perf_counter()
    
    # Test 1: Temperature profile
    temp_results = test_temperature_profile()
    
    # Test 2: Pressure-Volume
    pv_results = test_pressure_volume()
    
    # Test 3: Stress-Strain
    stress_strain_results = test_stress_strain()
    
    total_time = time.perf_counter() - total_start
    
    print("\n" + "=" * 70)
    print(f"Total execution time: {total_time:.3f} seconds")
    print("=" * 70)
    
    print("\n\nCONCLUSIONS:")
    print("-" * 70)
    print("1. Temperature Data (spatial distribution):")
    print("   - Cubic Spline provides smooth, physically realistic profiles")
    print("   - Least Squares good for smoothing noisy measurements")
    print("   - RBF flexible for irregular measurement points")
    print("\n2. Pressure-Volume (thermodynamic relationships):")
    print("   - Non-linear relationships require higher-order methods")
    print("   - Linear interpolation inadequate for PV = constant")
    print("   - Cubic Spline and RBF capture hyperbolic behavior well")
    print("\n3. Stress-Strain (material properties):")
    print("   - Cubic Spline and Hermite preserve C¹ continuity")
    print("   - Important for computing derived quantities (elastic modulus)")
    print("   - Smooth curves essential for material model fitting")
    print("\nGeneral Recommendations:")
    print("   - Use Cubic Spline as default for smooth engineering data")
    print("   - Use Least Squares when data has measurement noise")
    print("   - Use Hermite when derivative information available")
    print("   - Use RBF for scattered/irregular data points")


if __name__ == "__main__":
    main()