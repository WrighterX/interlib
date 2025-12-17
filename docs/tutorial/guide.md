# interlib Guide

This guide provides documentation across all interpolation modules in the Rust library.

## Module Header Documentation Template

Each module should start with comprehensive documentation following this pattern:

```rust
/// [Method Name] Interpolation Module
/// 
/// Brief description of the interpolation method and its purpose.
/// 
/// # Mathematical Background
/// 
/// Mathematical formulas and theory behind the method.
/// 
/// # Characteristics
/// 
/// - **Property 1**: Description
/// - **Property 2**: Description
/// - **Complexity**: Time/space complexity
/// 
/// # Use Cases
/// 
/// - Use case 1
/// - Use case 2
/// - When NOT to use
/// 
/// # Examples
/// 
/// ```python
/// from interlib import MethodInterpolator
/// 
/// # Usage example
/// ```
```

## Documentation Summary by Method

### 1. **Lagrange Interpolation** (`lagrange.rs`)
- **Type**: Global polynomial interpolation
- **Key Feature**: Exact fit through all points
- **Complexity**: O(n²) evaluation
- **Best For**: Small datasets (< 15 points)
- **Avoid**: Uniformly-spaced points (Runge's phenomenon)

### 2. **Newton Interpolation** (`newton.rs`)
- **Type**: Global polynomial interpolation (equivalent to Lagrange)
- **Key Feature**: Efficient divided differences computation
- **Complexity**: O(n²) setup, O(n) evaluation
- **Best For**: Repeated evaluations, incremental construction
- **Advantage**: Better performance than Lagrange

### 3. **Linear Interpolation** (`linear.rs`)
- **Type**: Piecewise linear
- **Key Feature**: Simple, fast, no oscillations
- **Complexity**: O(n) or O(log n) with binary search
- **Best For**: Simple data, fast computation needed
- **Limitation**: Not smooth (C⁰ continuous only)

### 4. **Quadratic Interpolation** (`quadratic.rs`)
- **Type**: Piecewise quadratic
- **Key Feature**: Captures curvature, smoother than linear
- **Complexity**: O(n) evaluation
- **Best For**: Moderately curved data
- **Note**: Uses overlapping triplets of points

### 5. **Cubic Spline** (`cubic_spline.rs`)
- **Type**: Piecewise cubic, C² continuous
- **Key Feature**: Smooth, industry standard
- **Complexity**: O(n) tridiagonal solve, O(log n) evaluation
- **Best For**: Smooth curves, engineering applications
- **Gold Standard**: Most commonly recommended method

### 6. **Hermite Interpolation** (`hermite.rs`)
- **Type**: Polynomial with derivative constraints
- **Key Feature**: Uses both values AND derivatives
- **Complexity**: O(n²) setup
- **Best For**: When derivative information available
- **Advantage**: More accurate with extra information

### 7. **Least Squares** (`least_squares.rs`)
- **Type**: Polynomial approximation (not exact fit)
- **Key Feature**: Smooths noisy data
- **Complexity**: O(nm²) where m is degree
- **Best For**: Noisy measurements, trend fitting
- **Provides**: R² goodness-of-fit metric

### 8. **RBF (Radial Basis Function)** (`rbf.rs`)
- **Type**: Global interpolation with basis functions
- **Key Features**: Multiple kernel types, handles scattered data
- **Kernels**: Gaussian, Multiquadric, Inverse Multiquadric, Thin Plate Spline, Linear
- **Complexity**: O(n²) for weights, O(n) evaluation
- **Best For**: Scattered data, multidimensional problems
- **Tunable**: Epsilon parameter controls smoothness

### 9. **Chebyshev Interpolation** (`chebyshev.rs`)
- **Type**: Polynomial with optimal node placement
- **Key Feature**: Avoids Runge's phenomenon
- **Complexity**: O(n log n) FFT-based
- **Best For**: High-accuracy function approximation
- **Advantage**: Spectral convergence for smooth functions
- **Uses**: Clenshaw algorithm for stable evaluation

## Python API Documentation Pattern

All interpolator classes follow this consistent API:

### Constructor
```python
interpolator = MethodInterpolator([parameters])
```

### Fitting
```python
interpolator.fit(x, y)  # Most methods
interpolator.fit(x, y, dy)  # Hermite (with derivatives)
```

### Evaluation
```python
# Single point
result = interpolator(x_value)

# Multiple points
results = interpolator([x1, x2, x3])
```

### Additional Methods (where applicable)
```python
coeffs = interpolator.get_coefficients()  # Newton, Hermite, Least Squares, Chebyshev
r_squared = interpolator.r_squared()  # Least Squares
nodes = interpolator.get_nodes()  # Chebyshev
```

## Quick Selection Guide

### Choose based on your needs:

| Requirement | Recommended Method |
|-------------|-------------------|
| Exact fit through all points | Lagrange, Newton, Cubic Spline |
| Smooth curves | Cubic Spline, RBF |
| Noisy data | Least Squares |
| Fast computation | Linear, Quadratic |
| High accuracy on smooth functions | Chebyshev, Cubic Spline |
| Derivative information available | Hermite |
| Avoid oscillations | Cubic Spline, Chebyshev, RBF |
| Scattered/irregular data | RBF, Cubic Spline |
| Small datasets (< 10 points) | Any method |
| Large datasets (> 100 points) | Linear, Cubic Spline |

## Common Pitfalls

### Runge's Phenomenon
- **Problem**: High-degree polynomials oscillate near boundaries
- **Affected**: Lagrange, Newton with uniform points
- **Solution**: Use Chebyshev nodes or piecewise methods (Spline)

### Noisy Data
- **Problem**: Exact interpolation amplifies noise
- **Affected**: All exact interpolation methods
- **Solution**: Use Least Squares or increase epsilon in RBF

### Extrapolation
- **Caution**: All methods can be unreliable outside data range
- **Best**: Cubic Spline (natural boundary conditions)
- **Worst**: High-degree polynomials (wild oscillations)

### Computational Cost
- **Expensive**: RBF for large n (O(n³) system solve)
- **Efficient**: Linear (O(n)), Cubic Spline (O(n))
- **Consider**: Pre-compute for repeated evaluations

## Error Messages Documentation

All methods provide clear error messages for:
- Not fitted: "Interpolator not fitted. Call fit(x, y) first."
- Length mismatch: "x and y must have the same length"
- Empty data: "x and y cannot be empty"
- Insufficient points: Method-specific minimum requirements
- Invalid input: "Input must be a float or a list of floats"

## Testing Recommendations

Document that each method should be tested with:
1. Simple cases (linear, quadratic functions)
2. Edge cases (2 points, many points)
3. Known functions (sin, cos, exp)
4. Problematic cases (Runge's function)
5. Noisy data
6. Boundary behavior

## Performance Benchmarks

Include typical performance metrics:
- Setup time (fitting)
- Evaluation time (single point)
- Evaluation time (1000 points)
- Memory usage
- Numerical accuracy