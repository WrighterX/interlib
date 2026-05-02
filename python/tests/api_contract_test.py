from interlib import (
    ChebyshevInterpolator,
    CubicSplineInterpolator,
    HermiteInterpolator,
    LagrangeInterpolator,
    LeastSquaresInterpolator,
    LinearInterpolator,
    NewtonInterpolator,
    QuadraticInterpolator,
    RBFInterpolator,
)


def assert_close(actual, expected, tol=1e-9):
    assert abs(actual - expected) <= tol, f"Expected {expected}, got {actual}"


def assert_float(value):
    assert isinstance(value, float), f"Expected float, got {type(value)}"


def assert_float_list(value, length):
    assert isinstance(value, list), f"Expected list, got {type(value)}"
    assert len(value) == length, f"Expected length {length}, got {len(value)}"
    assert all(isinstance(item, float) for item in value), value


def assert_raises_value_error(func):
    try:
        func()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def check_scalar_and_sequence(interp, scalar_x, vector_x):
    scalar = interp(scalar_x)
    vector = interp(vector_x)
    assert_float(scalar)
    assert_float_list(vector, len(vector_x))


PUBLIC_CLASSES = [
    ChebyshevInterpolator,
    CubicSplineInterpolator,
    HermiteInterpolator,
    LagrangeInterpolator,
    LeastSquaresInterpolator,
    LinearInterpolator,
    NewtonInterpolator,
    QuadraticInterpolator,
    RBFInterpolator,
]


def test_top_level_imports_and_constructors():
    for cls in PUBLIC_CLASSES:
        interp = cls()
        assert cls.__name__ in repr(interp)

    assert_raises_value_error(lambda: ChebyshevInterpolator(n_points=0))
    assert_raises_value_error(lambda: ChebyshevInterpolator(x_min=1.0, x_max=1.0))
    assert_raises_value_error(lambda: RBFInterpolator(kernel="unknown"))
    assert_raises_value_error(lambda: RBFInterpolator(epsilon=0.0))


def test_unfitted_operations_raise_value_error():
    for cls in PUBLIC_CLASSES:
        assert_raises_value_error(lambda cls=cls: cls()(0.0))

    assert_raises_value_error(lambda: NewtonInterpolator().get_coefficients())
    assert_raises_value_error(lambda: HermiteInterpolator().get_coefficients())
    assert_raises_value_error(lambda: LeastSquaresInterpolator().get_coefficients())
    assert_raises_value_error(lambda: ChebyshevInterpolator().get_coefficients())
    assert_raises_value_error(lambda: CubicSplineInterpolator().num_segments())
    assert_raises_value_error(lambda: LeastSquaresInterpolator().r_squared())
    assert_raises_value_error(lambda: RBFInterpolator().get_weights())


def test_scalar_and_sequence_evaluation_contracts():
    linear = LinearInterpolator()
    linear.fit([2.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    check_scalar_and_sequence(linear, 0.5, [0.0, 0.5, 1.0])
    assert_close(linear(0.5), 0.5)
    assert_close(linear(-1.0), 0.0)  # documented boundary-value extrapolation
    assert_raises_value_error(lambda: LinearInterpolator().fit([0.0, 0.0], [1.0, 2.0]))

    lagrange = LagrangeInterpolator()
    lagrange.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    check_scalar_and_sequence(lagrange, 1.5, [0.0, 1.0, 1.5])
    assert_close(lagrange(1.0), 1.0)

    newton = NewtonInterpolator()
    newton.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    check_scalar_and_sequence(newton, 1.5, [0.0, 1.0, 1.5])
    assert_close(newton(2.0), 4.0)

    quadratic = QuadraticInterpolator()
    quadratic.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    check_scalar_and_sequence(quadratic, 1.5, [0.0, 1.0, 1.5])
    assert_close(quadratic(2.0), 4.0)

    cubic = CubicSplineInterpolator()
    cubic.fit([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0])
    check_scalar_and_sequence(cubic, 1.5, [0.0, 1.0, 1.5])
    assert_close(cubic(2.0), 4.0)

    hermite = HermiteInterpolator()
    hermite.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], [0.0, 2.0, 4.0])
    check_scalar_and_sequence(hermite, 1.5, [0.0, 1.0, 1.5])
    assert_close(hermite(1.0), 1.0)

    least_squares = LeastSquaresInterpolator(degree=2)
    least_squares.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    check_scalar_and_sequence(least_squares, 1.5, [0.0, 1.0, 1.5])
    assert_close(least_squares(2.0), 4.0)

    rbf = RBFInterpolator(kernel="gaussian", epsilon=1.0)
    rbf.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    check_scalar_and_sequence(rbf, 1.5, [0.0, 1.0, 1.5])
    assert_close(rbf(1.0), 1.0, tol=1e-7)

    cheb = ChebyshevInterpolator(n_points=5, x_min=0.0, x_max=2.0)
    nodes = cheb.get_nodes()
    cheb.fit([x * x for x in nodes])
    check_scalar_and_sequence(cheb, 1.0, [0.0, 1.0, 2.0])
    assert_close(cheb(1.0), 1.0, tol=1e-7)
    assert_raises_value_error(lambda: cheb(-0.1))


def test_helper_methods():
    linear = LinearInterpolator()
    linear.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    linear.update_y([0.0, 2.0, 6.0])
    assert_close(linear(0.5), 1.0)
    linear.add_point(3.0, 9.0)
    assert_close(linear(3.0), 9.0)

    lagrange = LagrangeInterpolator()
    lagrange.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    lagrange.update_y([1.0, 2.0, 5.0])
    assert_close(lagrange(1.0), 2.0)
    lagrange.add_point(3.0, 10.0)
    assert_close(lagrange(3.0), 10.0)

    newton = NewtonInterpolator()
    newton.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert_float_list(newton.get_coefficients(), 3)

    hermite = HermiteInterpolator()
    hermite.fit([0.0, 1.0], [0.0, 1.0], [0.0, 2.0])
    assert_float_list(hermite.get_coefficients(), 4)

    least_squares = LeastSquaresInterpolator(degree=2)
    least_squares.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert least_squares.get_degree() == 2
    assert_float_list(least_squares.get_coefficients(), 3)
    assert_float(least_squares.r_squared())

    cubic = CubicSplineInterpolator()
    cubic.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert cubic.num_segments() == 2

    rbf = RBFInterpolator()
    rbf.fit([0.0, 1.0], [0.0, 1.0])
    assert_float_list(rbf.get_weights(), 2)

    cheb = ChebyshevInterpolator(n_points=4, x_min=0.0, x_max=1.0)
    assert_float_list(cheb.get_nodes(), 4)
    cheb.fit_function(lambda x: x * x)
    assert_float_list(cheb.get_coefficients(), 4)
    before = cheb(0.5)
    cheb.set_method(False)
    after = cheb(0.5)
    assert_close(before, after, tol=1e-7)


if __name__ == "__main__":
    test_top_level_imports_and_constructors()
    test_unfitted_operations_raise_value_error()
    test_scalar_and_sequence_evaluation_contracts()
    test_helper_methods()
    print("API contract smoke test passed")
