# interlib
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit)

interlib is an open-source Python library for interpolation methods designed as an alternative to scipy.interpolate. Being built on Rust, it provides reliable and in some cases faster solutions to unknown data point problems. It includes polynomial, piecewise, approximation-based and advanced interpolators for all of your needs.

## Usage
All of the library's built-in functions work out of the box, so the only thing you have to do is to import interlib into your project. In this case we will be using this Git repository directly.

> [!NOTE]
>
> At this development stage there's no way to install interlib except this Git repository.

Go to a directory where you want to install the library, and use `git clone` to clone the repository into your system.

When you are in the root folder of interlib, activate virtual environment via `venv`, and install needed dependencies:

```bash
pip install .
```

Finally, you build the project:

```bash
maturin develop
```

Congrats, you installed interlib into your system!

Now, there are several use cases for it:

- Numerical calculations for real-world problems.
- Engineering datasets interpolation (e.g. temperature data).
- Signal reconstruction from sampled data.
- Etc...

Equip yourself for whatever you have at hand.

## Examples
To import the library into your project, include the following line:

```python
import interlib
```

Or, if you want to import a concrete method, you can use the following:

```python
from interlib import LagrangeInterpolator
```

The ways to use the library's methods are generally the same across all of them. Let's take `LagrangeInterpolator` as an example. First, we have to create an instance:

```python
interp = LagrangeInterpolator()
```

Then we have to define known data points to fit them into the said instance:

```python
x = [1.0, 2.0, 3.0]
y = [1.0, 4.0, 9.0]
interp.fit(x, y)
```

Done! Now we can get the value at any point x. The `LagrangeInterpolator`, like most of the interlib's methods, can have scalar values *or* lists as parameters:

```python
# Evaluate at a single point
unknown_y = interp(4.0)
print(f"Interpolated value at x=4: {unknown_y}")

# Evaluate at multiple points
x_new = [1.5, 2.5, 3.5, 4.0]
y_new = interp(x_new)
print(f"Interpolated values: {y_new}")
```

at your option and use case.