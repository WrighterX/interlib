"""
A high-performance Python interpolation library implemented in Rust. Provides various interpolators like LinearInterpolator, QuadraticInterpolator, RBFInterpolator, etc.
"""

import builtins
import typing

@typing.final
class ChebyshevInterpolator:
    r"""
    Chebyshev Polynomial Interpolator
    
    Interpolator using Chebyshev polynomial expansion with optimal nodes.
    
    # Attributes
    
    * `x_min`, `x_max` - Interval boundaries
    * `nodes` - Pre-computed Chebyshev nodes
    * `y_values` - Function values at nodes
    * `coefficients` - Chebyshev expansion coefficients
    * `n_points` - Number of interpolation points
    * `use_clenshaw` - Whether to use Clenshaw algorithm
    * `fitted` - Whether coefficients have been computed
    """
    def __new__(cls, n_points: builtins.int = 10, x_min: builtins.float = -1.0, x_max: builtins.float = 1.0, use_clenshaw: builtins.bool = True) -> ChebyshevInterpolator:
        r"""
        Create a new Chebyshev interpolator
        
        Parameters
        ----------
        n_points : int, default=10
            Number of Chebyshev nodes (polynomial degree = n_points - 1)
        x_min : float, default=-1.0
            Left endpoint of interval
        x_max : float, default=1.0
            Right endpoint of interval
        use_clenshaw : bool, default=True
            Whether to use Clenshaw algorithm (recommended)
        
        Returns
        -------
        ChebyshevInterpolator
            A new, unfitted interpolator instance
        
        Raises
        ------
        ValueError
            If n_points is 0 or x_min >= x_max
        """
    def get_nodes(self) -> builtins.list[builtins.float]:
        r"""
        Get the Chebyshev nodes
        
        Returns
        -------
        list of float
            Optimally-placed Chebyshev nodes in [x_min, x_max]
        
        Notes
        -----
        These are the points where the function must be evaluated
        before calling fit().
        """
    def fit(self, y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit with function values at Chebyshev nodes
        
        Parameters
        ----------
        y : list of float
            Function values at the Chebyshev nodes
        
        Raises
        ------
        ValueError
            If y length doesn't match number of nodes
        """
    def fit_function(self, func: typing.Any) -> None:
        r"""
        Fit using a Python function
        
        Convenience method that evaluates the function at Chebyshev nodes
        automatically.
        
        Parameters
        ----------
        func : callable
            Python function to interpolate
        
        Raises
        ------
        ValueError
            If function call fails
        """
    def get_coefficients(self) -> builtins.list[builtins.float]:
        r"""
        Get Chebyshev polynomial coefficients
        
        Returns
        -------
        list of float
            Coefficients [c₀, c₁, ..., cₙ₋₁] for P(x) = Σ cₖTₖ(x)
        
        Raises
        ------
        ValueError
            If not fitted
        """
    def set_method(self, use_clenshaw: builtins.bool) -> None:
        r"""
        Set evaluation method
        
        Parameters
        ----------
        use_clenshaw : bool
            If True, use Clenshaw algorithm (stable, O(n))
            If False, use direct evaluation (simple, O(n²))
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate (must be in [x_min, x_max])
        
        Returns
        -------
        float or list of float
            Interpolated value(s)
        
        Raises
        ------
        ValueError
            If not fitted or x outside [x_min, x_max]
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation
        """

@typing.final
class CubicSplineInterpolator:
    r"""
    Natural Cubic Spline Interpolator
    
    A stateful interpolator that pre-computes cubic spline segments
    for smooth C² continuous interpolation.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of data points
    * `y_values` - Stored y coordinates of data points
    * `segments` - Pre-computed cubic polynomial segments
    * `fitted` - Whether the interpolator has been fitted
    """
    def __new__(cls) -> CubicSplineInterpolator:
        r"""
        Create a new cubic spline interpolator
        
        Returns
        -------
        CubicSplineInterpolator
            A new, unfitted interpolator instance
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the interpolator with data points
        
        Computes the natural cubic spline segments. Natural boundary conditions
        are used: second derivative equals zero at both endpoints.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points (must be strictly increasing)
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths
            If fewer than 2 data points are provided
            If x values are not strictly increasing
        
        Notes
        -----
        The spline segments are computed using the Thomas algorithm to solve
        the tridiagonal system for second derivatives, which is O(n) efficient.
        """
    def num_segments(self) -> builtins.int:
        r"""
        Get the number of spline segments
        
        Returns
        -------
        int
            Number of cubic polynomial segments (n_points - 1)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate the interpolation
        
        Returns
        -------
        float or list of float
            Interpolated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
            If input is neither a float nor a list of floats
        
        Notes
        -----
        For points outside the data range, the edge segments are used
        for extrapolation (linear extrapolation from the edge cubic).
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the interpolator
        
        Returns
        -------
        str
            Description of the interpolator state including number of segments
        """

@typing.final
class HermiteInterpolator:
    r"""
    Hermite Polynomial Interpolator
    
    A stateful interpolator that uses both function values and derivatives
    to construct a smooth polynomial interpolation.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of data points
    * `y_values` - Stored y coordinates (function values)
    * `dy_values` - Stored derivatives at data points
    * `z_values` - Doubled x values for divided differences
    * `coefficients` - Pre-computed Hermite coefficients
    * `fitted` - Whether the interpolator has been fitted
    """
    def __new__(cls) -> HermiteInterpolator:
        r"""
        Create a new Hermite interpolator
        
        Returns
        -------
        HermiteInterpolator
            A new, unfitted interpolator instance
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float], dy: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the interpolator with data points and derivatives
        
        Computes the Hermite polynomial coefficients using divided differences
        with doubled points.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points
        y : list of float
            Y coordinates (function values) at data points
        dy : list of float
            Derivatives (dy/dx) at data points
        
        Raises
        ------
        ValueError
            If x, y, and dy don't all have the same length
            If any of the arrays is empty
        
        Notes
        -----
        The quality of interpolation depends on the accuracy of the derivative
        values. Inaccurate derivatives can lead to poor interpolation results.
        
        Examples
        --------
        >>> import math
        >>> interp = HermiteInterpolator()
        >>> x = [0.0, 1.0, 2.0]
        >>> y = [0.0, 1.0, 8.0]  # x³
        >>> dy = [0.0, 3.0, 12.0]  # 3x²
        >>> interp.fit(x, y, dy)
        """
    def get_coefficients(self) -> builtins.list[builtins.float]:
        r"""
        Get the Hermite polynomial coefficients
        
        Returns the divided difference coefficients used in the Newton form
        of the Hermite polynomial.
        
        Returns
        -------
        list of float
            Hermite polynomial coefficients in Newton form
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
        
        Notes
        -----
        The returned coefficients are for the Newton form with doubled points.
        There are 2n coefficients for n data points.
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Uses Horner's method for efficient and stable evaluation.
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate the interpolation
        
        Returns
        -------
        float or list of float
            Hermite interpolated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
            If input is neither a float nor a list of floats
        
        Notes
        -----
        The Hermite polynomial passes through all data points and has the
        correct derivative at each point, providing very accurate interpolation
        for smooth functions.
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the interpolator
        
        Returns
        -------
        str
            Description of the interpolator state
        """

@typing.final
class LagrangeInterpolator:
    r"""
    Lagrange Polynomial Interpolator
    
    A stateful interpolator that fits a Lagrange polynomial through data points
    and allows evaluation at arbitrary points.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of fitted data points
    * `y_values` - Stored y coordinates of fitted data points
    * `fitted` - Whether the interpolator has been fitted with data
    """
    def __new__(cls) -> LagrangeInterpolator:
        r"""
        Create a new Lagrange interpolator
        
        Returns
        -------
        LagrangeInterpolator
            A new, unfitted interpolator instance
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the interpolator with data points
        
        Stores the data points for later evaluation. The Lagrange polynomial
        is constructed implicitly and evaluated on-demand.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths or if either is empty
        
        Examples
        --------
        >>> interp = LagrangeInterpolator()
        >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate the interpolation
        
        Returns
        -------
        float or list of float
            Interpolated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
            If input is neither a float nor a list of floats
        
        Examples
        --------
        >>> interp = LagrangeInterpolator()
        >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        >>> interp(1.5)  # Single point
        2.25
        >>> interp([0.5, 1.5])  # Multiple points
        [0.25, 2.25]
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the interpolator
        
        Returns
        -------
        str
            Description of the interpolator state
        """

@typing.final
class LeastSquaresInterpolator:
    r"""
    Least Squares Polynomial Approximator
    
    Fits a polynomial of specified degree that minimizes sum of squared errors.
    Provides R² metric to assess quality of fit.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of data points
    * `y_values` - Stored y coordinates of data points
    * `coefficients` - Computed polynomial coefficients
    * `degree` - Degree of the polynomial
    * `fitted` - Whether the approximator has been fitted
    """
    def __new__(cls, degree: builtins.int = 2) -> LeastSquaresInterpolator:
        r"""
        Create a new least squares approximator
        
        Parameters
        ----------
        degree : int, default=2
            Degree of polynomial to fit
        
            Guidelines:
            - degree=1: Linear regression
            - degree=2-3: Most common for noisy data
            - degree=4-6: For more complex patterns
            - degree > 10: Usually not recommended (overfitting)
        
        Returns
        -------
        LeastSquaresInterpolator
            A new, unfitted approximator instance
        
        Examples
        --------
        >>> ls = LeastSquaresInterpolator(degree=2)  # Quadratic fit
        >>> ls = LeastSquaresInterpolator(degree=1)  # Linear fit
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the polynomial to data points
        
        Computes the least squares solution that minimizes Σ(yᵢ - P(xᵢ))².
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths
            If fewer than degree+1 points are provided
            If fitting fails (singular matrix)
        
        Notes
        -----
        Requires at least degree+1 data points. More points improve stability
        and allow the method to smooth noise effectively.
        """
    def get_coefficients(self) -> builtins.list[builtins.float]:
        r"""
        Get the polynomial coefficients
        
        Returns coefficients in ascending degree order: [c₀, c₁, c₂, ...]
        representing the polynomial c₀ + c₁x + c₂x² + ...
        
        Returns
        -------
        list of float
            Polynomial coefficients [constant, linear, quadratic, ...]
        
        Raises
        ------
        ValueError
            If the approximator has not been fitted
        
        Examples
        --------
        >>> ls.fit([0, 1, 2], [1, 3, 4])
        >>> coeffs = ls.get_coefficients()
        >>> print(f"y = {coeffs[0]:.2f} + {coeffs[1]:.2f}x")
        """
    def get_degree(self) -> builtins.int:
        r"""
        Get the polynomial degree
        
        Returns
        -------
        int
            Degree of the fitted polynomial
        """
    def r_squared(self) -> builtins.float:
        r"""
        Compute R-squared (coefficient of determination)
        
        R² measures the proportion of variance in y explained by the model.
        
        Returns
        -------
        float
            R² value between 0 and 1
            - 1.0: Perfect fit (all variance explained)
            - 0.9: Very good fit (90% variance explained)
            - 0.5: Moderate fit
            - 0.0: Model no better than mean
            - <0.0: Model worse than mean (rare)
        
        Raises
        ------
        ValueError
            If the approximator has not been fitted
        
        Notes
        -----
        R² = 1 - (SS_res / SS_tot) where:
        - SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
        - SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the polynomial at one or more points
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate
        
        Returns
        -------
        float or list of float
            Approximated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the approximator has not been fitted
            If input is neither a float nor a list of floats
        
        Notes
        -----
        Can safely extrapolate, but extrapolation quality depends on
        polynomial degree and data distribution.
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the approximator
        
        Returns
        -------
        str
            Description including degree, number of points, and R² if fitted
        """

@typing.final
class LinearInterpolator:
    r"""
    Linear Interpolator
    
    A stateful interpolator that performs piecewise linear interpolation
    through data points.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of data points
    * `y_values` - Stored y coordinates of data points
    * `fitted` - Whether the interpolator has been fitted with data
    """
    def __new__(cls) -> LinearInterpolator:
        r"""
        Create a new linear interpolator
        
        Returns
        -------
        LinearInterpolator
            A new, unfitted interpolator instance
        
        Examples
        --------
        >>> interp = LinearInterpolator()
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the interpolator with data points
        
        Stores the data points for later evaluation. No pre-computation is needed
        for linear interpolation.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points (must be strictly increasing)
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths
            If x or y is empty
            If x values are not strictly increasing
        
        Notes
        -----
        X values must be sorted in strictly increasing order. This is verified
        during fitting to ensure correct interpolation behavior.
        
        Examples
        --------
        >>> interp = LinearInterpolator()
        >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate the interpolation
        
        Returns
        -------
        float or list of float
            Linearly interpolated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
            If input is neither a float nor a list of floats
        
        Notes
        -----
        For points outside the data range, the interpolator returns the
        nearest boundary value (constant extrapolation).
        
        Examples
        --------
        >>> interp = LinearInterpolator()
        >>> interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        >>> interp(0.5)  # Midpoint between 0 and 1
        0.5
        >>> interp([0.5, 1.5])  # Multiple points
        [0.5, 2.5]
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the interpolator
        
        Returns
        -------
        str
            Description of the interpolator state
        """

@typing.final
class NewtonInterpolator:
    r"""
    Newton Divided Differences Interpolator
    
    A stateful interpolator that pre-computes divided difference coefficients
    for efficient repeated evaluation.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of data points
    * `y_values` - Stored y coordinates of data points
    * `coefficients` - Pre-computed divided difference coefficients
    * `fitted` - Whether the interpolator has been fitted
    """
    def __new__(cls) -> NewtonInterpolator:
        r"""
        Create a new Newton interpolator
        
        Returns
        -------
        NewtonInterpolator
            A new, unfitted interpolator instance
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the interpolator with data points
        
        Computes and stores the divided difference coefficients for the Newton
        polynomial. This allows for efficient repeated evaluation.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths or if either is empty
        
        Notes
        -----
        The divided differences are computed once during fitting, making
        subsequent evaluations more efficient than Lagrange interpolation.
        """
    def get_coefficients(self) -> builtins.list[builtins.float]:
        r"""
        Get the Newton polynomial divided difference coefficients
        
        Returns
        -------
        list of float
            Divided difference coefficients [f[x₀], f[x₀,x₁], f[x₀,x₁,x₂], ...]
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
        
        Notes
        -----
        The i-th coefficient multiplies the term (x-x₀)(x-x₁)...(x-xᵢ₋₁)
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Uses Horner's method for efficient and numerically stable evaluation.
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate the interpolation
        
        Returns
        -------
        float or list of float
            Interpolated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
            If input is neither a float nor a list of floats
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the interpolator
        
        Returns
        -------
        str
            Description of the interpolator state
        """

@typing.final
class QuadraticInterpolator:
    r"""
    Piecewise Quadratic Interpolator
    
    A stateful interpolator that performs piecewise quadratic interpolation
    using overlapping triplets of data points.
    
    # Attributes
    
    * `x_values` - Stored x coordinates of data points
    * `y_values` - Stored y coordinates of data points
    * `fitted` - Whether the interpolator has been fitted
    """
    def __new__(cls) -> QuadraticInterpolator:
        r"""
        Create a new quadratic interpolator
        
        Returns
        -------
        QuadraticInterpolator
            A new, unfitted interpolator instance
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the interpolator with data points
        
        Stores the data points. Quadratic segments are computed on-demand
        during evaluation.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points (must be strictly increasing)
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths
            If fewer than 3 data points are provided
            If x values are not strictly increasing
        
        Notes
        -----
        Requires at least 3 data points to fit quadratics. For 2 points,
        falls back to linear interpolation.
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate the interpolation
        
        Returns
        -------
        float or list of float
            Quadratically interpolated value(s) at the specified point(s)
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
            If input is neither a float nor a list of floats
        
        Notes
        -----
        For each evaluation point, selects the nearest triplet of data points
        and fits a quadratic through them.
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation of the interpolator
        
        Returns
        -------
        str
            Description of the interpolator state
        """

@typing.final
class RBFInterpolator:
    r"""
    Radial Basis Function Interpolator
    
    Global interpolator using weighted radial basis functions.
    
    # Attributes
    
    * `x_values` - Stored data point locations
    * `y_values` - Stored function values
    * `weights` - Computed interpolation weights
    * `kernel` - Selected RBF kernel
    * `epsilon` - Shape parameter
    * `fitted` - Whether weights have been computed
    """
    def __new__(cls, kernel: builtins.str = 'gaussian', epsilon: builtins.float = 1.0) -> RBFInterpolator:
        r"""
        Create a new RBF interpolator
        
        Parameters
        ----------
        kernel : str, default="gaussian"
            RBF kernel type. Options:
            - "gaussian": Smooth, general purpose
            - "multiquadric": Robust, most common
            - "inverse_multiquadric": Very smooth
            - "thin_plate_spline": Minimal bending
            - "linear": Simple, for debugging
        epsilon : float, default=1.0
            Shape parameter (not used for thin_plate_spline, linear)
            - Smaller values → flatter, smoother
            - Larger values → sharper, more local
            - Typical range: 0.1 to 10
        
        Returns
        -------
        RBFInterpolator
            A new, unfitted interpolator instance
        
        Raises
        ------
        ValueError
            If kernel name is invalid or epsilon is non-positive
        """
    def fit(self, x: typing.Sequence[builtins.float], y: typing.Sequence[builtins.float]) -> None:
        r"""
        Fit the RBF interpolator
        
        Computes interpolation weights by solving the RBF system Φw = y.
        
        Parameters
        ----------
        x : list of float
            X coordinates of data points
        y : list of float
            Y coordinates of data points
        
        Raises
        ------
        ValueError
            If x and y have different lengths or are empty
            If weight computation fails (singular matrix)
        
        Notes
        -----
        Complexity is O(n³) due to solving the linear system.
        For large datasets (n > 1000), consider other methods.
        """
    def get_weights(self) -> builtins.list[builtins.float]:
        r"""
        Get the RBF interpolation weights
        
        Returns
        -------
        list of float
            Weight vector [w₀, w₁, ..., wₙ₋₁]
        
        Raises
        ------
        ValueError
            If the interpolator has not been fitted
        """
    def __call__(self, x: typing.Any) -> typing.Any:
        r"""
        Evaluate the interpolation at one or more points
        
        Parameters
        ----------
        x : float or list of float
            Point(s) at which to evaluate
        
        Returns
        -------
        float or list of float
            Interpolated value(s)
        
        Raises
        ------
        ValueError
            If not fitted or invalid input type
        """
    def __repr__(self) -> builtins.str:
        r"""
        String representation
        """

