use std::hint::black_box;
use std::time::Instant;

use super::chebyshev_core::ChebyshevCore;
use super::core_trait::InterpolationCore;
use super::cubic_spline_core::CubicSplineCore;
use super::hermite_core::HermiteCore;
use super::lagrange_core::LagrangeCore;
use super::least_squares_core::LeastSquaresCore;
use super::linear_core::LinearCore;
use super::newton_core::NewtonCore;
use super::quadratic_core::QuadraticCore;
use super::rbf_core::{RBFCore, RBFKernel};

const WARMUP: usize = 1;
const RUNS: usize = 5;

struct BenchResult {
    name: &'static str,
    median_ms: f64,
    checksum: f64,
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

fn training_values(xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .map(|&x| (0.7 * x).sin() + 0.15 * (1.9 * x).cos())
        .collect()
}

fn derivative_values(xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .map(|&x| 0.7 * (0.7 * x).cos() - 0.285 * (1.9 * x).sin())
        .collect()
}

fn shuffled_eval_points(n: usize, start: f64, end: f64) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let scale = end - start;
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        values.push(start + scale * unit);
    }
    values
}

fn sampled_sum(values: &[f64]) -> f64 {
    let step = (values.len() / 97).max(1);
    let mut sum = 0.0;
    let mut i = 0;
    while i < values.len() {
        sum += values[i];
        i += step;
    }
    black_box(sum)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn time_case<F>(mut f: F) -> (f64, f64)
where
    F: FnMut() -> f64,
{
    let mut samples = Vec::with_capacity(RUNS);
    let mut checksum = 0.0;
    for run in 0..(WARMUP + RUNS) {
        let start = Instant::now();
        let sample_sum = f();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(
            sample_sum.is_finite(),
            "benchmark checksum became non-finite"
        );
        if run >= WARMUP {
            samples.push(elapsed_ms);
            checksum += sample_sum;
        }
    }
    (median(samples), checksum)
}

fn bench_linear() -> BenchResult {
    let x = linspace(0.0, 10.0, 4096);
    let y = training_values(&x);
    let xs_sorted = linspace(-0.5, 10.5, 160_000);
    let xs_mixed = shuffled_eval_points(40_000, -0.5, 10.5);
    let mut out_sorted = vec![0.0; xs_sorted.len()];
    let mut out_mixed = vec![0.0; xs_mixed.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = LinearCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs_sorted), black_box(&mut out_sorted))
            .unwrap();
        core.fill_many(black_box(&xs_mixed), black_box(&mut out_mixed))
            .unwrap();
        sampled_sum(&out_sorted) + sampled_sum(&out_mixed)
    });

    BenchResult {
        name: "linear",
        median_ms,
        checksum,
    }
}

fn bench_quadratic() -> BenchResult {
    let x = linspace(0.0, 10.0, 2048);
    let y = training_values(&x);
    let xs_sorted = linspace(-0.5, 10.5, 120_000);
    let xs_mixed = shuffled_eval_points(30_000, -0.5, 10.5);
    let mut out_sorted = vec![0.0; xs_sorted.len()];
    let mut out_mixed = vec![0.0; xs_mixed.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = QuadraticCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs_sorted), black_box(&mut out_sorted))
            .unwrap();
        core.fill_many(black_box(&xs_mixed), black_box(&mut out_mixed))
            .unwrap();
        sampled_sum(&out_sorted) + sampled_sum(&out_mixed)
    });

    BenchResult {
        name: "quadratic",
        median_ms,
        checksum,
    }
}

fn bench_cubic_spline() -> BenchResult {
    let x = linspace(0.0, 10.0, 2048);
    let y = training_values(&x);
    let xs_sorted = linspace(-0.5, 10.5, 120_000);
    let xs_mixed = shuffled_eval_points(30_000, -0.5, 10.5);
    let mut out_sorted = vec![0.0; xs_sorted.len()];
    let mut out_mixed = vec![0.0; xs_mixed.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = CubicSplineCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs_sorted), black_box(&mut out_sorted))
            .unwrap();
        core.fill_many(black_box(&xs_mixed), black_box(&mut out_mixed))
            .unwrap();
        sampled_sum(&out_sorted) + sampled_sum(&out_mixed)
    });

    BenchResult {
        name: "cubic_spline",
        median_ms,
        checksum,
    }
}

fn bench_lagrange() -> BenchResult {
    let x = linspace(-1.0, 1.0, 48);
    let y = training_values(&x);
    let xs = linspace(-1.0, 1.0, 35_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = LagrangeCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "lagrange",
        median_ms,
        checksum,
    }
}

fn bench_newton() -> BenchResult {
    let x = linspace(-1.0, 1.0, 48);
    let y = training_values(&x);
    let xs = linspace(-1.0, 1.0, 35_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = NewtonCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "newton",
        median_ms,
        checksum,
    }
}

fn bench_hermite() -> BenchResult {
    let x = linspace(-1.0, 1.0, 36);
    let y = training_values(&x);
    let dy = derivative_values(&x);
    let xs = linspace(-1.0, 1.0, 30_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = HermiteCore::new();
        core.fit(
            black_box(x.clone()),
            black_box(y.clone()),
            black_box(dy.clone()),
        )
        .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "hermite",
        median_ms,
        checksum,
    }
}

fn bench_least_squares() -> BenchResult {
    let x = linspace(-1.0, 1.0, 4096);
    let y = training_values(&x);
    let xs = linspace(-1.0, 1.0, 180_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = LeastSquaresCore::new(5);
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "least_squares",
        median_ms,
        checksum,
    }
}

fn bench_chebyshev() -> BenchResult {
    let xs = linspace(-1.0, 1.0, 180_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = ChebyshevCore::new(128, -1.0, 1.0, true).unwrap();
        let nodes = core.nodes().to_vec();
        let y = training_values(&nodes);
        core.fit(black_box(&y)).unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "chebyshev",
        median_ms,
        checksum,
    }
}

fn bench_rbf() -> BenchResult {
    let x = linspace(-1.0, 1.0, 32);
    let y = training_values(&x);
    let xs = linspace(-1.0, 1.0, 12_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = RBFCore::new(RBFKernel::Gaussian, 3.0).unwrap();
        core.fit(black_box(&x), black_box(&y)).unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "rbf",
        median_ms,
        checksum,
    }
}

fn bench_linear_broad() -> BenchResult {
    let x_small = linspace(0.0, 10.0, 32);
    let y_small = training_values(&x_small);
    let xs_small = linspace(-0.5, 10.5, 4_096);
    let mut out_small = vec![0.0; xs_small.len()];

    let x_large = linspace(0.0, 10.0, 16_384);
    let y_large = training_values(&x_large);
    let xs_large_sorted = linspace(-0.5, 10.5, 240_000);
    let xs_large_mixed = shuffled_eval_points(60_000, -0.5, 10.5);
    let mut out_large_sorted = vec![0.0; xs_large_sorted.len()];
    let mut out_large_mixed = vec![0.0; xs_large_mixed.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut small = LinearCore::new();
        small
            .fit(black_box(x_small.clone()), black_box(y_small.clone()))
            .unwrap();
        small
            .fill_many(black_box(&xs_small), black_box(&mut out_small))
            .unwrap();

        let mut large = LinearCore::new();
        large
            .fit(black_box(x_large.clone()), black_box(y_large.clone()))
            .unwrap();
        large
            .fill_many(
                black_box(&xs_large_sorted),
                black_box(&mut out_large_sorted),
            )
            .unwrap();
        large
            .fill_many(black_box(&xs_large_mixed), black_box(&mut out_large_mixed))
            .unwrap();

        sampled_sum(&out_small) + sampled_sum(&out_large_sorted) + sampled_sum(&out_large_mixed)
    });

    BenchResult {
        name: "linear_broad",
        median_ms,
        checksum,
    }
}

fn bench_quadratic_broad() -> BenchResult {
    let x_small = linspace(0.0, 10.0, 32);
    let y_small = training_values(&x_small);
    let xs_small = linspace(-0.5, 10.5, 4_096);
    let mut out_small = vec![0.0; xs_small.len()];

    let x_large = linspace(0.0, 10.0, 8_192);
    let y_large = training_values(&x_large);
    let xs_large_sorted = linspace(-0.5, 10.5, 180_000);
    let xs_large_mixed = shuffled_eval_points(45_000, -0.5, 10.5);
    let mut out_large_sorted = vec![0.0; xs_large_sorted.len()];
    let mut out_large_mixed = vec![0.0; xs_large_mixed.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut small = QuadraticCore::new();
        small
            .fit(black_box(x_small.clone()), black_box(y_small.clone()))
            .unwrap();
        small
            .fill_many(black_box(&xs_small), black_box(&mut out_small))
            .unwrap();

        let mut large = QuadraticCore::new();
        large
            .fit(black_box(x_large.clone()), black_box(y_large.clone()))
            .unwrap();
        large
            .fill_many(
                black_box(&xs_large_sorted),
                black_box(&mut out_large_sorted),
            )
            .unwrap();
        large
            .fill_many(black_box(&xs_large_mixed), black_box(&mut out_large_mixed))
            .unwrap();

        sampled_sum(&out_small) + sampled_sum(&out_large_sorted) + sampled_sum(&out_large_mixed)
    });

    BenchResult {
        name: "quadratic_broad",
        median_ms,
        checksum,
    }
}

fn bench_cubic_spline_broad() -> BenchResult {
    let x_small = linspace(0.0, 10.0, 32);
    let y_small = training_values(&x_small);
    let xs_small = linspace(-0.5, 10.5, 4_096);
    let mut out_small = vec![0.0; xs_small.len()];

    let x_large = linspace(0.0, 10.0, 8_192);
    let y_large = training_values(&x_large);
    let xs_large_sorted = linspace(-0.5, 10.5, 180_000);
    let xs_large_mixed = shuffled_eval_points(45_000, -0.5, 10.5);
    let mut out_large_sorted = vec![0.0; xs_large_sorted.len()];
    let mut out_large_mixed = vec![0.0; xs_large_mixed.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut small = CubicSplineCore::new();
        small
            .fit(black_box(x_small.clone()), black_box(y_small.clone()))
            .unwrap();
        small
            .fill_many(black_box(&xs_small), black_box(&mut out_small))
            .unwrap();

        let mut large = CubicSplineCore::new();
        large
            .fit(black_box(x_large.clone()), black_box(y_large.clone()))
            .unwrap();
        large
            .fill_many(
                black_box(&xs_large_sorted),
                black_box(&mut out_large_sorted),
            )
            .unwrap();
        large
            .fill_many(black_box(&xs_large_mixed), black_box(&mut out_large_mixed))
            .unwrap();

        sampled_sum(&out_small) + sampled_sum(&out_large_sorted) + sampled_sum(&out_large_mixed)
    });

    BenchResult {
        name: "cubic_spline_broad",
        median_ms,
        checksum,
    }
}

fn bench_lagrange_medium() -> BenchResult {
    let x = linspace(-1.0, 1.0, 128);
    let y = training_values(&x);
    let xs = linspace(-1.0, 1.0, 12_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = LagrangeCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "lagrange_medium",
        median_ms,
        checksum,
    }
}

fn bench_newton_medium() -> BenchResult {
    let x = linspace(-1.0, 1.0, 128);
    let y = training_values(&x);
    let xs = linspace(-1.0, 1.0, 12_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = NewtonCore::new();
        core.fit(black_box(x.clone()), black_box(y.clone()))
            .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "newton_medium",
        median_ms,
        checksum,
    }
}

fn bench_hermite_medium() -> BenchResult {
    let x = linspace(-1.0, 1.0, 64);
    let y = training_values(&x);
    let dy = derivative_values(&x);
    let xs = linspace(-1.0, 1.0, 12_000);
    let mut out = vec![0.0; xs.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut core = HermiteCore::new();
        core.fit(
            black_box(x.clone()),
            black_box(y.clone()),
            black_box(dy.clone()),
        )
        .unwrap();
        core.fill_many(black_box(&xs), black_box(&mut out)).unwrap();
        sampled_sum(&out)
    });

    BenchResult {
        name: "hermite_medium",
        median_ms,
        checksum,
    }
}

fn bench_least_squares_broad() -> BenchResult {
    let x_small = linspace(-1.0, 1.0, 64);
    let y_small = training_values(&x_small);
    let xs_small = linspace(-1.0, 1.0, 4_096);
    let mut out_small = vec![0.0; xs_small.len()];

    let x_large = linspace(-1.0, 1.0, 16_384);
    let y_large = training_values(&x_large);
    let xs_large = linspace(-1.0, 1.0, 240_000);
    let mut out_large = vec![0.0; xs_large.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut small = LeastSquaresCore::new(5);
        small
            .fit(black_box(x_small.clone()), black_box(y_small.clone()))
            .unwrap();
        small
            .fill_many(black_box(&xs_small), black_box(&mut out_small))
            .unwrap();

        let mut large = LeastSquaresCore::new(5);
        large
            .fit(black_box(x_large.clone()), black_box(y_large.clone()))
            .unwrap();
        large
            .fill_many(black_box(&xs_large), black_box(&mut out_large))
            .unwrap();

        sampled_sum(&out_small) + sampled_sum(&out_large)
    });

    BenchResult {
        name: "least_squares_broad",
        median_ms,
        checksum,
    }
}

fn bench_chebyshev_broad() -> BenchResult {
    let xs_small = linspace(-1.0, 1.0, 4_096);
    let mut out_small = vec![0.0; xs_small.len()];
    let xs_large = linspace(-1.0, 1.0, 160_000);
    let mut out_large = vec![0.0; xs_large.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut small = ChebyshevCore::new(32, -1.0, 1.0, true).unwrap();
        let small_nodes = small.nodes().to_vec();
        let small_y = training_values(&small_nodes);
        small.fit(black_box(&small_y)).unwrap();
        small
            .fill_many(black_box(&xs_small), black_box(&mut out_small))
            .unwrap();

        let mut large = ChebyshevCore::new(1_024, -1.0, 1.0, true).unwrap();
        let large_nodes = large.nodes().to_vec();
        let large_y = training_values(&large_nodes);
        large.fit(black_box(&large_y)).unwrap();
        large
            .fill_many(black_box(&xs_large), black_box(&mut out_large))
            .unwrap();

        sampled_sum(&out_small) + sampled_sum(&out_large)
    });

    BenchResult {
        name: "chebyshev_broad",
        median_ms,
        checksum,
    }
}

fn bench_rbf_broad() -> BenchResult {
    let x_small = linspace(-1.0, 1.0, 32);
    let y_small = training_values(&x_small);
    let xs_small = linspace(-1.0, 1.0, 4_096);
    let mut out_small = vec![0.0; xs_small.len()];

    let x_medium = linspace(-1.0, 1.0, 256);
    let y_medium = training_values(&x_medium);
    let xs_medium = linspace(-1.0, 1.0, 12_000);
    let mut out_medium = vec![0.0; xs_medium.len()];

    let (median_ms, checksum) = time_case(|| {
        let mut small = RBFCore::new(RBFKernel::Gaussian, 3.0).unwrap();
        small.fit(black_box(&x_small), black_box(&y_small)).unwrap();
        small
            .fill_many(black_box(&xs_small), black_box(&mut out_small))
            .unwrap();

        let mut medium = RBFCore::new(RBFKernel::Gaussian, 40.0).unwrap();
        medium
            .fit(black_box(&x_medium), black_box(&y_medium))
            .unwrap();
        medium
            .fill_many(black_box(&xs_medium), black_box(&mut out_medium))
            .unwrap();

        sampled_sum(&out_small) + sampled_sum(&out_medium)
    });

    BenchResult {
        name: "rbf_broad",
        median_ms,
        checksum,
    }
}

#[test]
#[ignore = "autoresearch broad small/large performance harness; run via ./autoresearch.sh"]
fn core_perf_harness_broad_metrics() {
    let results = [
        bench_linear_broad(),
        bench_quadratic_broad(),
        bench_cubic_spline_broad(),
        bench_lagrange_medium(),
        bench_newton_medium(),
        bench_hermite_medium(),
        bench_least_squares_broad(),
        bench_chebyshev_broad(),
        bench_rbf_broad(),
    ];

    let total_ms: f64 = results.iter().map(|result| result.median_ms).sum();
    let checksum: f64 = results.iter().map(|result| result.checksum).sum();
    assert!(
        checksum.is_finite(),
        "combined broad benchmark checksum became non-finite"
    );

    println!("METRIC broad_core_total_ms={total_ms:.6}");
    println!("METRIC checksum={checksum:.12}");
    for result in &results {
        println!("METRIC {}_ms={:.6}", result.name, result.median_ms);
    }
}

#[test]
#[ignore = "autoresearch performance harness; run via ./autoresearch.sh"]
fn core_perf_harness_metrics() {
    let results = [
        bench_linear(),
        bench_quadratic(),
        bench_cubic_spline(),
        bench_lagrange(),
        bench_newton(),
        bench_hermite(),
        bench_least_squares(),
        bench_chebyshev(),
        bench_rbf(),
    ];

    let total_ms: f64 = results.iter().map(|result| result.median_ms).sum();
    let checksum: f64 = results.iter().map(|result| result.checksum).sum();
    assert!(
        checksum.is_finite(),
        "combined benchmark checksum became non-finite"
    );

    println!("METRIC core_total_ms={total_ms:.6}");
    println!("METRIC checksum={checksum:.12}");
    for result in &results {
        println!("METRIC {}_ms={:.6}", result.name, result.median_ms);
    }
}
