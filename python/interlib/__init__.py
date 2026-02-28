"""
Interlib: High-performance Python interpolation library built with Rust.

Provides 9 interpolation methods as fast alternatives to scipy.interpolate.
"""

from .interlib import (
    LagrangeInterpolator,
    NewtonInterpolator,
    LinearInterpolator,
    QuadraticInterpolator,
    CubicSplineInterpolator,
    HermiteInterpolator,
    LeastSquaresInterpolator,
    RBFInterpolator,
    ChebyshevInterpolator,
)

__all__ = [
    "LagrangeInterpolator",
    "NewtonInterpolator",
    "LinearInterpolator",
    "QuadraticInterpolator",
    "CubicSplineInterpolator",
    "HermiteInterpolator",
    "LeastSquaresInterpolator",
    "RBFInterpolator",
    "ChebyshevInterpolator",
]
