"""
Focused core benchmark for safe-core-performance work.

Compares interlib vs SciPy for:
- Linear
- Cubic Spline
- Hermite (vs CubicHermiteSpline)
- RBF Gaussian

This is a local evidence harness (not a CI benchmark).
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline as ScipyCubicSpline
from scipy.interpolate import RBFInterpolator as ScipyRBFInterpolator
from scipy.interpolate import interp1d

from interlib import (
    CubicSplineInterpolator,
    HermiteInterpolator,
    LinearInterpolator,
    RBFInterpolator,
)


@dataclass
class MethodSpec:
    name: str
    train_sizes: list[int]
    n_eval: int


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def run_once_linear(n_train: int, n_eval: int):
    x_train = np.linspace(0.0, 10.0, n_train)
    y_train = np.sin(x_train)
    x_eval = np.linspace(0.0, 10.0, n_eval)

    # interlib
    il = LinearInterpolator()
    t0 = time.perf_counter()
    il.fit(x_train.tolist(), y_train.tolist())
    t1 = time.perf_counter()
    _ = il(x_eval.tolist())
    t2 = time.perf_counter()

    # scipy
    t3 = time.perf_counter()
    sp = interp1d(x_train, y_train, kind="linear")
    t4 = time.perf_counter()
    _ = sp(x_eval)
    t5 = time.perf_counter()

    return (t1 - t0, t2 - t1), (t4 - t3, t5 - t4)


def run_once_cubicspline(n_train: int, n_eval: int):
    x_train = np.linspace(0.0, 10.0, n_train)
    y_train = np.sin(x_train)
    x_eval = np.linspace(0.0, 10.0, n_eval)

    il = CubicSplineInterpolator()
    t0 = time.perf_counter()
    il.fit(x_train.tolist(), y_train.tolist())
    t1 = time.perf_counter()
    _ = il(x_eval.tolist())
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    sp = ScipyCubicSpline(x_train, y_train)
    t4 = time.perf_counter()
    _ = sp(x_eval)
    t5 = time.perf_counter()

    return (t1 - t0, t2 - t1), (t4 - t3, t5 - t4)


def run_once_hermite(n_train: int, n_eval: int):
    x_train = np.linspace(0.0, 10.0, n_train)
    y_train = np.sin(x_train)
    dy_train = np.cos(x_train)
    x_eval = np.linspace(0.0, 10.0, n_eval)

    il = HermiteInterpolator()
    t0 = time.perf_counter()
    il.fit(x_train.tolist(), y_train.tolist(), dy_train.tolist())
    t1 = time.perf_counter()
    _ = il(x_eval.tolist())
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    sp = CubicHermiteSpline(x_train, y_train, dy_train)
    t4 = time.perf_counter()
    _ = sp(x_eval)
    t5 = time.perf_counter()

    return (t1 - t0, t2 - t1), (t4 - t3, t5 - t4)


def run_once_rbf(n_train: int, n_eval: int):
    x_train = np.linspace(0.0, 10.0, n_train)
    y_train = np.sin(x_train)
    x_eval = np.linspace(0.0, 10.0, n_eval)

    il = RBFInterpolator(kernel="gaussian", epsilon=1.0)
    t0 = time.perf_counter()
    il.fit(x_train.tolist(), y_train.tolist())
    t1 = time.perf_counter()
    _ = il(x_eval.tolist())
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    sp = ScipyRBFInterpolator(x_train.reshape(-1, 1), y_train, kernel="gaussian", epsilon=1.0)
    t4 = time.perf_counter()
    _ = sp(x_eval.reshape(-1, 1))
    t5 = time.perf_counter()

    return (t1 - t0, t2 - t1), (t4 - t3, t5 - t4)


RUNNERS = {
    "linear": run_once_linear,
    "cubicspline": run_once_cubicspline,
    "hermite": run_once_hermite,
    "rbf": run_once_rbf,
}

METHODS = {
    "linear": MethodSpec("Linear", [128, 512, 2048], 100_000),
    "cubicspline": MethodSpec("CubicSpline", [128, 512, 2048], 100_000),
    "hermite": MethodSpec("Hermite", [64, 128, 256], 100_000),
    "rbf": MethodSpec("RBF", [32, 64, 128], 40_000),
}


def benchmark_method(method_key: str, runs: int, warmup: int):
    spec = METHODS[method_key]
    runner = RUNNERS[method_key]

    print(f"\n=== {spec.name} ===")
    print(f"eval_points={spec.n_eval}")
    print("n_train | interlib_fit_ms interlib_eval_ms interlib_total_ms | scipy_fit_ms scipy_eval_ms scipy_total_ms | speedup_total")

    for n in spec.train_sizes:
        il_fit, il_eval = [], []
        sp_fit, sp_eval = [], []
        failed_reason = None

        for i in range(runs + warmup):
            try:
                (a_fit, a_eval), (b_fit, b_eval) = runner(n, spec.n_eval)
            except Exception as exc:
                failed_reason = str(exc)
                break

            if i >= warmup:
                il_fit.append(a_fit)
                il_eval.append(a_eval)
                sp_fit.append(b_fit)
                sp_eval.append(b_eval)

        if failed_reason is not None:
            print(f"{n:7d} | FAILED: {failed_reason}")
            continue

        il_fit_ms = median_ms(il_fit)
        il_eval_ms = median_ms(il_eval)
        sp_fit_ms = median_ms(sp_fit)
        sp_eval_ms = median_ms(sp_eval)
        il_total = il_fit_ms + il_eval_ms
        sp_total = sp_fit_ms + sp_eval_ms
        speedup = sp_total / il_total if il_total > 0 else 0.0

        print(
            f"{n:7d} | {il_fit_ms:14.3f} {il_eval_ms:15.3f} {il_total:16.3f} | "
            f"{sp_fit_ms:12.3f} {sp_eval_ms:13.3f} {sp_total:14.3f} | {speedup:12.3f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Focused benchmark for core performance changes")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--methods",
        type=str,
        default="linear,cubicspline,hermite,rbf",
        help="Comma-separated subset: linear,cubicspline,hermite,rbf",
    )
    args = parser.parse_args()

    selected = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    print("Focused interlib vs SciPy benchmark")
    print(f"runs={args.runs}, warmup={args.warmup}")

    for method_key in selected:
        if method_key not in METHODS:
            print(f"Skipping unknown method: {method_key}")
            continue
        benchmark_method(method_key, args.runs, args.warmup)


if __name__ == "__main__":
    main()
