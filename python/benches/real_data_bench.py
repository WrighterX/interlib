"""
Real-data benchmark suite for interlib vs scipy.interpolate.

This benchmark uses cached snapshots of official public datasets instead of
synthetic functions. Supported sources:

- NOAA / NWS observations API
- NASA / JPL Horizons vectors API

The download step is intentionally outside the timed benchmark loop.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import (
    BarycentricInterpolator,
    CubicSpline as ScipyCubicSpline,
    CubicHermiteSpline,
    RBFInterpolator as ScipyRBFInterpolator,
    interp1d,
)

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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _datasets import RealSeries, load_nasa_horizons_vectors, load_noaa_observations


@dataclass(slots=True)
class MethodSpec:
    name: str
    display_name: str
    build_interlib: Callable[[np.ndarray, np.ndarray], Any]
    eval_interlib: Callable[[Any, np.ndarray], np.ndarray]
    build_baseline: Callable[[np.ndarray, np.ndarray], Any] | None
    eval_baseline: Callable[[Any, np.ndarray], np.ndarray] | None
    baseline_label: str | None
    default_sizes: list[int] | None


@dataclass(slots=True)
class BenchmarkResult:
    method: str
    display_name: str
    train_size: int
    interlib_fit_ms: float
    interlib_eval_ms: float
    interlib_rmse: float
    interlib_max_error: float
    baseline_label: str | None
    baseline_fit_ms: float | None
    baseline_eval_ms: float | None
    baseline_rmse: float | None
    baseline_max_error: float | None
    interlib_warning: str | None
    baseline_warning: str | None


def _interleave_holdout_split(
    x: np.ndarray,
    y: np.ndarray,
    train_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_size < 4:
        raise ValueError("train_size must be at least 4")
    if len(x) <= train_size:
        raise ValueError("dataset is too small for the requested train_size")

    train_idx = np.unique(np.linspace(0, len(x) - 1, train_size, dtype=int))
    mask = np.ones(len(x), dtype=bool)
    mask[train_idx] = False

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[mask]
    y_test = y[mask]

    if len(x_test) == 0:
        raise ValueError("no held-out samples available for evaluation")

    return x_train, y_train, x_test, y_test


def _finite_difference_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _max_error(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(pred - truth)))


def _result_warning(pred: np.ndarray) -> str | None:
    if not np.all(np.isfinite(pred)):
        return "non-finite predictions"
    if pred.size and float(np.max(np.abs(pred))) > 1e100:
        return "extreme magnitude predictions"
    return None


def _numpy_eval(interpolator: Any, x_test: np.ndarray) -> np.ndarray:
    return np.asarray(interpolator(x_test), dtype=float)


def _build_hermite(x_train: np.ndarray, y_train: np.ndarray) -> HermiteInterpolator:
    interpolator = HermiteInterpolator()
    interpolator.fit(x_train, y_train, _finite_difference_derivatives(x_train, y_train))
    return interpolator


def _build_scipy_hermite(x_train: np.ndarray, y_train: np.ndarray) -> CubicHermiteSpline:
    return CubicHermiteSpline(x_train, y_train, _finite_difference_derivatives(x_train, y_train))


def _build_rbf(x_train: np.ndarray, y_train: np.ndarray) -> RBFInterpolator:
    interpolator = RBFInterpolator(kernel="gaussian", epsilon=1.0)
    interpolator.fit(x_train, y_train)
    return interpolator


def _build_chebyshev(x_train: np.ndarray, y_train: np.ndarray) -> ChebyshevInterpolator:
    interpolator = ChebyshevInterpolator(
        n_points=len(x_train),
        x_min=float(x_train[0]),
        x_max=float(x_train[-1]),
        use_clenshaw=True,
    )
    nodes = np.asarray(interpolator.get_nodes(), dtype=float)
    node_values = np.interp(nodes, x_train, y_train)
    interpolator.fit(node_values)
    return interpolator


def _build_least_squares(x_train: np.ndarray, y_train: np.ndarray) -> LeastSquaresInterpolator:
    degree = min(5, max(2, len(x_train) // 6))
    interpolator = LeastSquaresInterpolator(degree=degree)
    interpolator.fit(x_train, y_train)
    return interpolator


def _build_numpy_least_squares(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    degree = min(5, max(2, len(x_train) // 6))
    return np.polynomial.Polynomial.fit(x_train, y_train, deg=degree)


def _build_numpy_chebyshev(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    degree = max(1, len(x_train) - 1)
    return np.polynomial.chebyshev.Chebyshev.fit(
        x_train,
        y_train,
        deg=degree,
        domain=[float(x_train[0]), float(x_train[-1])],
    )


METHODS: dict[str, MethodSpec] = {
    "lagrange": MethodSpec(
        name="lagrange",
        display_name="lagrange",
        build_interlib=lambda x, y: _fit_simple(LagrangeInterpolator, x, y),
        eval_interlib=_numpy_eval,
        build_baseline=lambda x, y: BarycentricInterpolator(x, y),
        eval_baseline=_numpy_eval,
        baseline_label="scipy",
        default_sizes=None,
    ),
    "newton": MethodSpec(
        name="newton",
        display_name="newton",
        build_interlib=lambda x, y: _fit_simple(NewtonInterpolator, x, y),
        eval_interlib=_numpy_eval,
        build_baseline=lambda x, y: BarycentricInterpolator(x, y),
        eval_baseline=_numpy_eval,
        baseline_label="scipy",
        default_sizes=None,
    ),
    "linear": MethodSpec(
        name="linear",
        display_name="linear",
        build_interlib=lambda x, y: _fit_simple(LinearInterpolator, x, y),
        eval_interlib=_numpy_eval,
        build_baseline=lambda x, y: interp1d(x, y, kind="linear"),
        eval_baseline=_numpy_eval,
        baseline_label="scipy",
        default_sizes=None,
    ),
    "quadratic": MethodSpec(
        name="quadratic",
        display_name="quadratic",
        build_interlib=lambda x, y: _fit_simple(QuadraticInterpolator, x, y),
        eval_interlib=_numpy_eval,
        build_baseline=lambda x, y: interp1d(x, y, kind="quadratic"),
        eval_baseline=_numpy_eval,
        baseline_label="scipy",
        default_sizes=None,
    ),
    "cubicspline": MethodSpec(
        name="cubicspline",
        display_name="cubicspline",
        build_interlib=lambda x, y: _fit_simple(CubicSplineInterpolator, x, y),
        eval_interlib=_numpy_eval,
        build_baseline=lambda x, y: ScipyCubicSpline(x, y),
        eval_baseline=_numpy_eval,
        baseline_label="scipy",
        default_sizes=None,
    ),
    "hermite": MethodSpec(
        name="hermite",
        display_name="hermite",
        build_interlib=_build_hermite,
        eval_interlib=_numpy_eval,
        build_baseline=_build_scipy_hermite,
        eval_baseline=_numpy_eval,
        baseline_label="scipy",
        default_sizes=None,
    ),
    "leastsquares": MethodSpec(
        name="leastsquares",
        display_name="leastsquares",
        build_interlib=_build_least_squares,
        eval_interlib=_numpy_eval,
        build_baseline=_build_numpy_least_squares,
        eval_baseline=_numpy_eval,
        baseline_label="numpy",
        default_sizes=None,
    ),
    "rbf": MethodSpec(
        name="rbf",
        display_name="rbf",
        build_interlib=_build_rbf,
        eval_interlib=_numpy_eval,
        build_baseline=lambda x, y: ScipyRBFInterpolator(x.reshape(-1, 1), y, kernel="gaussian", epsilon=1.0),
        eval_baseline=lambda interpolator, x_test: np.asarray(interpolator(x_test.reshape(-1, 1)), dtype=float),
        baseline_label="scipy",
        default_sizes=None,
    ),
    "chebyshev": MethodSpec(
        name="chebyshev",
        display_name="chebyshev*",
        build_interlib=_build_chebyshev,
        eval_interlib=_numpy_eval,
        build_baseline=_build_numpy_chebyshev,
        eval_baseline=_numpy_eval,
        baseline_label="numpy*",
        default_sizes=None,
    ),
}


def _fit_simple(cls: type, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    interpolator = cls()
    interpolator.fit(x_train, y_train)
    return interpolator


def _time_runs(
    builder: Callable[[np.ndarray, np.ndarray], Any],
    evaluator: Callable[[Any, np.ndarray], np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_runs: int,
    n_warmup: int,
) -> tuple[float, float, float, float, str | None]:
    fit_samples: list[float] = []
    eval_samples: list[float] = []
    predictions: np.ndarray | None = None

    gc.collect()
    gc.disable()
    try:
        for run_idx in range(n_runs + n_warmup):
            start = time.perf_counter()
            interpolator = builder(x_train, y_train)
            fit_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            predictions = evaluator(interpolator, x_test)
            eval_elapsed = time.perf_counter() - start

            if run_idx >= n_warmup:
                fit_samples.append(fit_elapsed)
                eval_samples.append(eval_elapsed)
    finally:
        gc.enable()

    if predictions is None:
        raise RuntimeError("benchmark produced no predictions")

    warning = _result_warning(predictions)
    rmse = _rmse(predictions, y_test)
    max_error = _max_error(predictions, y_test)
    return _median_ms(fit_samples), _median_ms(eval_samples), rmse, max_error, warning


def benchmark_method(
    method: MethodSpec,
    series: RealSeries,
    train_size: int,
    *,
    n_runs: int,
    n_warmup: int,
) -> BenchmarkResult:
    x = np.asarray(series.x, dtype=float)
    y = np.asarray(series.y, dtype=float)
    x_train, y_train, x_test, y_test = _interleave_holdout_split(x, y, train_size)

    il_fit, il_eval, il_rmse, il_max_error, interlib_warning = _time_runs(
        method.build_interlib,
        method.eval_interlib,
        x_train,
        y_train,
        x_test,
        y_test,
        n_runs,
        n_warmup,
    )

    baseline_fit = None
    baseline_eval = None
    baseline_rmse = None
    baseline_max_error = None
    baseline_warning = None

    if method.build_baseline is not None and method.eval_baseline is not None:
        baseline_fit, baseline_eval, baseline_rmse, baseline_max_error, baseline_warning = _time_runs(
            method.build_baseline,
            method.eval_baseline,
            x_train,
            y_train,
            x_test,
            y_test,
            n_runs,
            n_warmup,
        )

    return BenchmarkResult(
        method=method.name,
        display_name=method.display_name,
        train_size=train_size,
        interlib_fit_ms=il_fit,
        interlib_eval_ms=il_eval,
        interlib_rmse=il_rmse,
        interlib_max_error=il_max_error,
        baseline_label=method.baseline_label,
        baseline_fit_ms=baseline_fit,
        baseline_eval_ms=baseline_eval,
        baseline_rmse=baseline_rmse,
        baseline_max_error=baseline_max_error,
        interlib_warning=interlib_warning,
        baseline_warning=baseline_warning,
    )


def resolve_series(args: argparse.Namespace) -> RealSeries:
    if args.dataset == "noaa":
        return load_noaa_observations(
            station_id=args.station,
            field=args.field,
            limit=args.limit,
            refresh=args.refresh,
        )
    if args.dataset == "nasa":
        return load_nasa_horizons_vectors(
            command=args.command,
            axis=args.axis,
            center=args.center,
            start_time=args.start_time,
            stop_time=args.stop_time,
            step_size=args.step_size,
            refresh=args.refresh,
        )
    raise ValueError(f"Unsupported dataset '{args.dataset}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark interlib with cached real public datasets")
    parser.add_argument("--dataset", choices=["noaa", "nasa"], required=True)
    parser.add_argument(
        "--methods",
        default="linear,cubicspline,rbf",
        help="Comma-separated method keys. Available: " + ", ".join(sorted(METHODS)),
    )
    parser.add_argument("--sizes", default="", help="Comma-separated train sizes. Defaults depend on method")
    parser.add_argument("--n-runs", type=int, default=9)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--refresh", action="store_true", help="Refresh the local cached dataset snapshot")

    parser.add_argument("--station", default="KSFO", help="NOAA station id")
    parser.add_argument(
        "--field",
        default="temperature",
        choices=["temperature", "dewpoint", "wind_speed", "pressure"],
        help="NOAA observation field",
    )
    parser.add_argument("--limit", type=int, default=240, help="Maximum NOAA observations to request")

    parser.add_argument("--command", default="499", help="NASA Horizons COMMAND id, e.g. 499 for Mars")
    parser.add_argument("--axis", default="x", choices=["x", "y", "z"], help="NASA vector axis to benchmark")
    parser.add_argument("--center", default="500@0", help="NASA Horizons CENTER value")
    parser.add_argument("--start-time", default="2026-01-01", help="NASA Horizons START_TIME")
    parser.add_argument("--stop-time", default="2026-02-01", help="NASA Horizons STOP_TIME")
    parser.add_argument("--step-size", default="1 d", help="NASA Horizons STEP_SIZE")
    return parser.parse_args()


def _dataset_relative_sizes(method_name: str, dataset_length: int) -> list[int]:
    if dataset_length < 8:
        return []

    if method_name in {"linear", "quadratic", "cubicspline"}:
        fractions = [0.1, 0.2, 0.4, 0.7]
        floor = 16
        ceiling = max(16, dataset_length - 1)
    elif method_name in {"leastsquares", "chebyshev", "hermite"}:
        fractions = [0.08, 0.15, 0.25, 0.4]
        floor = 12
        ceiling = min(256, dataset_length - 1)
    elif method_name == "rbf":
        fractions = [0.05, 0.1, 0.15, 0.25]
        floor = 8
        ceiling = min(128, dataset_length - 1)
    elif method_name in {"lagrange", "newton"}:
        fractions = [0.03, 0.05, 0.08, 0.12]
        floor = 8
        ceiling = min(32, dataset_length - 1)
    else:
        fractions = [0.1, 0.2, 0.4]
        floor = 8
        ceiling = max(8, dataset_length - 1)

    sizes = {
        min(ceiling, max(floor, int(round(dataset_length * fraction))))
        for fraction in fractions
    }
    return sorted(size for size in sizes if 4 <= size < dataset_length)


def _selected_sizes(method: MethodSpec, series: RealSeries, requested_sizes: list[int]) -> list[int]:
    dataset_length = len(series.x)
    max_size = dataset_length - 1
    if requested_sizes:
        source_sizes = requested_sizes
    elif method.default_sizes is not None:
        source_sizes = method.default_sizes
    else:
        source_sizes = _dataset_relative_sizes(method.name, dataset_length)
    return [size for size in source_sizes if 4 <= size <= max_size]


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _print_series_header(series: RealSeries) -> None:
    print(f"Dataset: {series.name}")
    print(f"Source:  {series.source}")
    print(f"URL:     {series.source_url}")
    print(f"Points:  {len(series.x)}")
    print(f"Units:   x={series.x_unit}, y={series.y_unit}")
    print(f"Note:    {series.description}")
    print()


def _print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.display_name:12} n={result.train_size:4d} | "
        f"interlib fit={result.interlib_fit_ms:8.3f} ms eval={result.interlib_eval_ms:8.3f} ms "
        f"rmse={result.interlib_rmse:10.6g} max={result.interlib_max_error:10.6g} | "
        f"{(result.baseline_label or 'baseline')} fit={_format_optional(result.baseline_fit_ms):>8} ms "
        f"eval={_format_optional(result.baseline_eval_ms):>8} ms "
        f"rmse={_format_optional(result.baseline_rmse):>8} "
        f"max={_format_optional(result.baseline_max_error):>8}"
    )
    if result.interlib_warning:
        print(f"{'':12} warning: interlib {result.interlib_warning}")
    if result.baseline_warning:
        print(f"{'':12} warning: {result.baseline_label or 'baseline'} {result.baseline_warning}")


def main() -> int:
    args = parse_args()
    requested_methods = [name.strip().lower() for name in args.methods.split(",") if name.strip()]
    requested_sizes = [int(item) for item in args.sizes.split(",") if item.strip()]

    unknown_methods = [name for name in requested_methods if name not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")

    series = resolve_series(args)
    _print_series_header(series)
    if "chebyshev" in requested_methods:
        print("Note: chebyshev* uses a NumPy approximation baseline and is not a strict apples-to-apples equivalent.")
        print()

    any_success = False

    for method_name in requested_methods:
        method = METHODS[method_name]
        sizes = _selected_sizes(method, series, requested_sizes)
        if not sizes:
            print(f"{method_name:12} skipped: no valid train sizes for dataset length {len(series.x)}")
            continue

        for train_size in sizes:
            try:
                result = benchmark_method(
                    method,
                    series,
                    train_size,
                    n_runs=args.n_runs,
                    n_warmup=args.n_warmup,
                )
            except Exception as exc:
                print(f"{method_name:12} n={train_size:4d} failed: {exc}")
                continue

            any_success = True
            _print_result(result)

    return 0 if any_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
