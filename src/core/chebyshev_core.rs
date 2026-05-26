use std::f64::consts::PI;

use crate::core::core_error::CoreError;
use crate::core::core_trait::InterpolationCore;

fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    let mut nodes = vec![0.0; n];
    let half_width = 0.5 * (b - a);
    let midpoint = 0.5 * (b + a);
    for k in 0..(n / 2) {
        let x = ((2 * k + 1) as f64 * PI / (2 * n) as f64).cos();
        nodes[k] = half_width * x + midpoint;
        nodes[n - 1 - k] = midpoint - half_width * x;
    }
    if n % 2 == 1 {
        nodes[n / 2] = midpoint;
    }
    nodes
}

fn transform_to_standard(x: f64, a: f64, b: f64) -> f64 {
    2.0 * (x - a) / (b - a) - 1.0
}

fn chebyshev_polynomial(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut t_n_minus_two = 1.0;
    let mut t_n_minus_one = x;
    let mut t_n = x;
    for _ in 2..=n {
        t_n = 2.0 * x * t_n_minus_one - t_n_minus_two;
        t_n_minus_two = t_n_minus_one;
        t_n_minus_one = t_n;
    }
    t_n
}

fn compute_chebyshev_coefficients(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let n_f = n as f64;
    let mut coefficients = vec![0.0; n];

    for j in 0..(n / 2) {
        let y_even = y[j] + y[n - 1 - j];
        let y_odd = y[j] - y[n - 1 - j];
        coefficients[0] += y_even;
        if n == 1 {
            continue;
        }

        let theta = (2.0 * j as f64 + 1.0) * PI / (2.0 * n_f);
        let cos_theta = theta.cos();
        let two_cos_theta = 2.0 * cos_theta;
        coefficients[1] += y_odd * cos_theta;

        if n > 2 {
            let cos2_theta = two_cos_theta * cos_theta - 1.0;
            let step_two = 2.0 * cos2_theta;
            let mut even_prev = 1.0;
            let mut even_cur = cos2_theta;
            coefficients[2] += y_even * even_cur;

            let mut odd_prev = cos_theta;
            let mut odd_cur = two_cos_theta * cos2_theta - cos_theta;
            if n > 3 {
                coefficients[3] += y_odd * odd_cur;
            }

            let mut k = 4;
            while k + 3 < n {
                let even_next = step_two * even_cur - even_prev;
                coefficients[k] += y_even * even_next;
                even_prev = even_cur;
                even_cur = even_next;

                let odd_next = step_two * odd_cur - odd_prev;
                coefficients[k + 1] += y_odd * odd_next;
                odd_prev = odd_cur;
                odd_cur = odd_next;

                let even_next = step_two * even_cur - even_prev;
                coefficients[k + 2] += y_even * even_next;
                even_prev = even_cur;
                even_cur = even_next;

                let odd_next = step_two * odd_cur - odd_prev;
                coefficients[k + 3] += y_odd * odd_next;
                odd_prev = odd_cur;
                odd_cur = odd_next;

                k += 4;
            }
            while k + 1 < n {
                let even_next = step_two * even_cur - even_prev;
                coefficients[k] += y_even * even_next;
                even_prev = even_cur;
                even_cur = even_next;

                let odd_next = step_two * odd_cur - odd_prev;
                coefficients[k + 1] += y_odd * odd_next;
                odd_prev = odd_cur;
                odd_cur = odd_next;

                k += 2;
            }
            if k < n {
                let even_next = step_two * even_cur - even_prev;
                coefficients[k] += y_even * even_next;
            }
        }
    }

    if n % 2 == 1 {
        let j = n / 2;
        let y_j = y[j];
        coefficients[0] += y_j;
        if n > 1 {
            let theta = (2.0 * j as f64 + 1.0) * PI / (2.0 * n_f);
            let cos_theta = theta.cos();
            let two_cos_theta = 2.0 * cos_theta;
            let mut cos_prev = 1.0;
            let mut cos_cur = cos_theta;
            coefficients[1] += y_j * cos_cur;

            for coefficient in coefficients.iter_mut().skip(2) {
                let cos_next = two_cos_theta * cos_cur - cos_prev;
                *coefficient += y_j * cos_next;
                cos_prev = cos_cur;
                cos_cur = cos_next;
            }
        }
    }

    coefficients[0] /= n_f;
    let scale = 2.0 / n_f;
    for coefficient in coefficients.iter_mut().skip(1) {
        *coefficient *= scale;
    }
    coefficients
}

fn chebyshev_evaluate_clenshaw(coefficients: &[f64], x_std: f64) -> f64 {
    let n = coefficients.len();
    let mut b_k_plus_two = 0.0;
    let mut b_k_plus_one = 0.0;
    for k in (0..n).rev() {
        let b_k = coefficients[k] + 2.0 * x_std * b_k_plus_one - b_k_plus_two;
        b_k_plus_two = b_k_plus_one;
        b_k_plus_one = b_k;
    }
    b_k_plus_one - x_std * b_k_plus_two
}

fn chebyshev_evaluate_clenshaw4(
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
) -> [f64; 4] {
    let n = coefficients.len();
    let x0_twice = 2.0 * x0;
    let x1_twice = 2.0 * x1;
    let x2_twice = 2.0 * x2;
    let x3_twice = 2.0 * x3;
    let mut b2_0 = 0.0;
    let mut b1_0 = 0.0;
    let mut b2_1 = 0.0;
    let mut b1_1 = 0.0;
    let mut b2_2 = 0.0;
    let mut b1_2 = 0.0;
    let mut b2_3 = 0.0;
    let mut b1_3 = 0.0;

    let mut k = n;
    while k > 1 {
        k -= 1;
        let coef = coefficients[k];

        let b0 = coef + x0_twice * b1_0 - b2_0;
        b2_0 = b1_0;
        b1_0 = b0;

        let b0 = coef + x1_twice * b1_1 - b2_1;
        b2_1 = b1_1;
        b1_1 = b0;

        let b0 = coef + x2_twice * b1_2 - b2_2;
        b2_2 = b1_2;
        b1_2 = b0;

        let b0 = coef + x3_twice * b1_3 - b2_3;
        b2_3 = b1_3;
        b1_3 = b0;

        k -= 1;
        let coef = coefficients[k];

        let b0 = coef + x0_twice * b1_0 - b2_0;
        b2_0 = b1_0;
        b1_0 = b0;

        let b0 = coef + x1_twice * b1_1 - b2_1;
        b2_1 = b1_1;
        b1_1 = b0;

        let b0 = coef + x2_twice * b1_2 - b2_2;
        b2_2 = b1_2;
        b1_2 = b0;

        let b0 = coef + x3_twice * b1_3 - b2_3;
        b2_3 = b1_3;
        b1_3 = b0;
    }
    if k == 1 {
        let coef = coefficients[0];

        let b0 = coef + x0_twice * b1_0 - b2_0;
        b2_0 = b1_0;
        b1_0 = b0;

        let b0 = coef + x1_twice * b1_1 - b2_1;
        b2_1 = b1_1;
        b1_1 = b0;

        let b0 = coef + x2_twice * b1_2 - b2_2;
        b2_2 = b1_2;
        b1_2 = b0;

        let b0 = coef + x3_twice * b1_3 - b2_3;
        b2_3 = b1_3;
        b1_3 = b0;
    }

    [
        b1_0 - x0 * b2_0,
        b1_1 - x1 * b2_1,
        b1_2 - x2 * b2_2,
        b1_3 - x3 * b2_3,
    ]
}

#[allow(clippy::too_many_arguments)]
fn chebyshev_evaluate_clenshaw8(
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    x5: f64,
    x6: f64,
    x7: f64,
) -> [f64; 8] {
    let n = coefficients.len();
    let x0_twice = 2.0 * x0;
    let x1_twice = 2.0 * x1;
    let x2_twice = 2.0 * x2;
    let x3_twice = 2.0 * x3;
    let x4_twice = 2.0 * x4;
    let x5_twice = 2.0 * x5;
    let x6_twice = 2.0 * x6;
    let x7_twice = 2.0 * x7;

    let mut b2_0 = 0.0;
    let mut b1_0 = 0.0;
    let mut b2_1 = 0.0;
    let mut b1_1 = 0.0;
    let mut b2_2 = 0.0;
    let mut b1_2 = 0.0;
    let mut b2_3 = 0.0;
    let mut b1_3 = 0.0;
    let mut b2_4 = 0.0;
    let mut b1_4 = 0.0;
    let mut b2_5 = 0.0;
    let mut b1_5 = 0.0;
    let mut b2_6 = 0.0;
    let mut b1_6 = 0.0;
    let mut b2_7 = 0.0;
    let mut b1_7 = 0.0;

    macro_rules! step {
        ($coef:expr) => {{
            let coef = $coef;

            let b0 = coef + x0_twice * b1_0 - b2_0;
            b2_0 = b1_0;
            b1_0 = b0;

            let b0 = coef + x1_twice * b1_1 - b2_1;
            b2_1 = b1_1;
            b1_1 = b0;

            let b0 = coef + x2_twice * b1_2 - b2_2;
            b2_2 = b1_2;
            b1_2 = b0;

            let b0 = coef + x3_twice * b1_3 - b2_3;
            b2_3 = b1_3;
            b1_3 = b0;

            let b0 = coef + x4_twice * b1_4 - b2_4;
            b2_4 = b1_4;
            b1_4 = b0;

            let b0 = coef + x5_twice * b1_5 - b2_5;
            b2_5 = b1_5;
            b1_5 = b0;

            let b0 = coef + x6_twice * b1_6 - b2_6;
            b2_6 = b1_6;
            b1_6 = b0;

            let b0 = coef + x7_twice * b1_7 - b2_7;
            b2_7 = b1_7;
            b1_7 = b0;
        }};
    }

    let mut k = n;
    while k > 1 {
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
    }
    if k == 1 {
        step!(coefficients[0]);
    }

    [
        b1_0 - x0 * b2_0,
        b1_1 - x1 * b2_1,
        b1_2 - x2 * b2_2,
        b1_3 - x3 * b2_3,
        b1_4 - x4 * b2_4,
        b1_5 - x5 * b2_5,
        b1_6 - x6 * b2_6,
        b1_7 - x7 * b2_7,
    ]
}

#[allow(clippy::too_many_arguments)]
fn chebyshev_evaluate_clenshaw16(
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    x5: f64,
    x6: f64,
    x7: f64,
    x8: f64,
    x9: f64,
    x10: f64,
    x11: f64,
    x12: f64,
    x13: f64,
    x14: f64,
    x15: f64,
) -> [f64; 16] {
    let n = coefficients.len();
    let x0_twice = 2.0 * x0;
    let x1_twice = 2.0 * x1;
    let x2_twice = 2.0 * x2;
    let x3_twice = 2.0 * x3;
    let x4_twice = 2.0 * x4;
    let x5_twice = 2.0 * x5;
    let x6_twice = 2.0 * x6;
    let x7_twice = 2.0 * x7;
    let x8_twice = 2.0 * x8;
    let x9_twice = 2.0 * x9;
    let x10_twice = 2.0 * x10;
    let x11_twice = 2.0 * x11;
    let x12_twice = 2.0 * x12;
    let x13_twice = 2.0 * x13;
    let x14_twice = 2.0 * x14;
    let x15_twice = 2.0 * x15;

    let mut b2_0 = 0.0;
    let mut b1_0 = 0.0;
    let mut b2_1 = 0.0;
    let mut b1_1 = 0.0;
    let mut b2_2 = 0.0;
    let mut b1_2 = 0.0;
    let mut b2_3 = 0.0;
    let mut b1_3 = 0.0;
    let mut b2_4 = 0.0;
    let mut b1_4 = 0.0;
    let mut b2_5 = 0.0;
    let mut b1_5 = 0.0;
    let mut b2_6 = 0.0;
    let mut b1_6 = 0.0;
    let mut b2_7 = 0.0;
    let mut b1_7 = 0.0;
    let mut b2_8 = 0.0;
    let mut b1_8 = 0.0;
    let mut b2_9 = 0.0;
    let mut b1_9 = 0.0;
    let mut b2_10 = 0.0;
    let mut b1_10 = 0.0;
    let mut b2_11 = 0.0;
    let mut b1_11 = 0.0;
    let mut b2_12 = 0.0;
    let mut b1_12 = 0.0;
    let mut b2_13 = 0.0;
    let mut b1_13 = 0.0;
    let mut b2_14 = 0.0;
    let mut b1_14 = 0.0;
    let mut b2_15 = 0.0;
    let mut b1_15 = 0.0;

    macro_rules! step {
        ($coef:expr) => {{
            let coef = $coef;

            let b0 = coef + x0_twice * b1_0 - b2_0;
            b2_0 = b1_0;
            b1_0 = b0;
            let b0 = coef + x1_twice * b1_1 - b2_1;
            b2_1 = b1_1;
            b1_1 = b0;
            let b0 = coef + x2_twice * b1_2 - b2_2;
            b2_2 = b1_2;
            b1_2 = b0;
            let b0 = coef + x3_twice * b1_3 - b2_3;
            b2_3 = b1_3;
            b1_3 = b0;
            let b0 = coef + x4_twice * b1_4 - b2_4;
            b2_4 = b1_4;
            b1_4 = b0;
            let b0 = coef + x5_twice * b1_5 - b2_5;
            b2_5 = b1_5;
            b1_5 = b0;
            let b0 = coef + x6_twice * b1_6 - b2_6;
            b2_6 = b1_6;
            b1_6 = b0;
            let b0 = coef + x7_twice * b1_7 - b2_7;
            b2_7 = b1_7;
            b1_7 = b0;
            let b0 = coef + x8_twice * b1_8 - b2_8;
            b2_8 = b1_8;
            b1_8 = b0;
            let b0 = coef + x9_twice * b1_9 - b2_9;
            b2_9 = b1_9;
            b1_9 = b0;
            let b0 = coef + x10_twice * b1_10 - b2_10;
            b2_10 = b1_10;
            b1_10 = b0;
            let b0 = coef + x11_twice * b1_11 - b2_11;
            b2_11 = b1_11;
            b1_11 = b0;
            let b0 = coef + x12_twice * b1_12 - b2_12;
            b2_12 = b1_12;
            b1_12 = b0;
            let b0 = coef + x13_twice * b1_13 - b2_13;
            b2_13 = b1_13;
            b1_13 = b0;
            let b0 = coef + x14_twice * b1_14 - b2_14;
            b2_14 = b1_14;
            b1_14 = b0;
            let b0 = coef + x15_twice * b1_15 - b2_15;
            b2_15 = b1_15;
            b1_15 = b0;
        }};
    }

    let mut k = n;
    while k > 7 {
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
    }
    while k > 0 {
        k -= 1;
        step!(coefficients[k]);
    }

    [
        b1_0 - x0 * b2_0,
        b1_1 - x1 * b2_1,
        b1_2 - x2 * b2_2,
        b1_3 - x3 * b2_3,
        b1_4 - x4 * b2_4,
        b1_5 - x5 * b2_5,
        b1_6 - x6 * b2_6,
        b1_7 - x7 * b2_7,
        b1_8 - x8 * b2_8,
        b1_9 - x9 * b2_9,
        b1_10 - x10 * b2_10,
        b1_11 - x11 * b2_11,
        b1_12 - x12 * b2_12,
        b1_13 - x13 * b2_13,
        b1_14 - x14 * b2_14,
        b1_15 - x15 * b2_15,
    ]
}

#[allow(clippy::too_many_arguments)]
fn chebyshev_evaluate_clenshaw32(
    coefficients: &[f64],
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
    x5: f64,
    x6: f64,
    x7: f64,
    x8: f64,
    x9: f64,
    x10: f64,
    x11: f64,
    x12: f64,
    x13: f64,
    x14: f64,
    x15: f64,
    x16: f64,
    x17: f64,
    x18: f64,
    x19: f64,
    x20: f64,
    x21: f64,
    x22: f64,
    x23: f64,
    x24: f64,
    x25: f64,
    x26: f64,
    x27: f64,
    x28: f64,
    x29: f64,
    x30: f64,
    x31: f64,
) -> [f64; 32] {
    let n = coefficients.len();
    let x0_twice = 2.0 * x0;
    let x1_twice = 2.0 * x1;
    let x2_twice = 2.0 * x2;
    let x3_twice = 2.0 * x3;
    let x4_twice = 2.0 * x4;
    let x5_twice = 2.0 * x5;
    let x6_twice = 2.0 * x6;
    let x7_twice = 2.0 * x7;
    let x8_twice = 2.0 * x8;
    let x9_twice = 2.0 * x9;
    let x10_twice = 2.0 * x10;
    let x11_twice = 2.0 * x11;
    let x12_twice = 2.0 * x12;
    let x13_twice = 2.0 * x13;
    let x14_twice = 2.0 * x14;
    let x15_twice = 2.0 * x15;
    let x16_twice = 2.0 * x16;
    let x17_twice = 2.0 * x17;
    let x18_twice = 2.0 * x18;
    let x19_twice = 2.0 * x19;
    let x20_twice = 2.0 * x20;
    let x21_twice = 2.0 * x21;
    let x22_twice = 2.0 * x22;
    let x23_twice = 2.0 * x23;
    let x24_twice = 2.0 * x24;
    let x25_twice = 2.0 * x25;
    let x26_twice = 2.0 * x26;
    let x27_twice = 2.0 * x27;
    let x28_twice = 2.0 * x28;
    let x29_twice = 2.0 * x29;
    let x30_twice = 2.0 * x30;
    let x31_twice = 2.0 * x31;

    let mut b2_0 = 0.0;
    let mut b1_0 = 0.0;
    let mut b2_1 = 0.0;
    let mut b1_1 = 0.0;
    let mut b2_2 = 0.0;
    let mut b1_2 = 0.0;
    let mut b2_3 = 0.0;
    let mut b1_3 = 0.0;
    let mut b2_4 = 0.0;
    let mut b1_4 = 0.0;
    let mut b2_5 = 0.0;
    let mut b1_5 = 0.0;
    let mut b2_6 = 0.0;
    let mut b1_6 = 0.0;
    let mut b2_7 = 0.0;
    let mut b1_7 = 0.0;
    let mut b2_8 = 0.0;
    let mut b1_8 = 0.0;
    let mut b2_9 = 0.0;
    let mut b1_9 = 0.0;
    let mut b2_10 = 0.0;
    let mut b1_10 = 0.0;
    let mut b2_11 = 0.0;
    let mut b1_11 = 0.0;
    let mut b2_12 = 0.0;
    let mut b1_12 = 0.0;
    let mut b2_13 = 0.0;
    let mut b1_13 = 0.0;
    let mut b2_14 = 0.0;
    let mut b1_14 = 0.0;
    let mut b2_15 = 0.0;
    let mut b1_15 = 0.0;
    let mut b2_16 = 0.0;
    let mut b1_16 = 0.0;
    let mut b2_17 = 0.0;
    let mut b1_17 = 0.0;
    let mut b2_18 = 0.0;
    let mut b1_18 = 0.0;
    let mut b2_19 = 0.0;
    let mut b1_19 = 0.0;
    let mut b2_20 = 0.0;
    let mut b1_20 = 0.0;
    let mut b2_21 = 0.0;
    let mut b1_21 = 0.0;
    let mut b2_22 = 0.0;
    let mut b1_22 = 0.0;
    let mut b2_23 = 0.0;
    let mut b1_23 = 0.0;
    let mut b2_24 = 0.0;
    let mut b1_24 = 0.0;
    let mut b2_25 = 0.0;
    let mut b1_25 = 0.0;
    let mut b2_26 = 0.0;
    let mut b1_26 = 0.0;
    let mut b2_27 = 0.0;
    let mut b1_27 = 0.0;
    let mut b2_28 = 0.0;
    let mut b1_28 = 0.0;
    let mut b2_29 = 0.0;
    let mut b1_29 = 0.0;
    let mut b2_30 = 0.0;
    let mut b1_30 = 0.0;
    let mut b2_31 = 0.0;
    let mut b1_31 = 0.0;

    macro_rules! step {
        ($coef:expr) => {{
            let coef = $coef;
            let b0 = coef + x0_twice * b1_0 - b2_0;
            b2_0 = b1_0;
            b1_0 = b0;
            let b0 = coef + x1_twice * b1_1 - b2_1;
            b2_1 = b1_1;
            b1_1 = b0;
            let b0 = coef + x2_twice * b1_2 - b2_2;
            b2_2 = b1_2;
            b1_2 = b0;
            let b0 = coef + x3_twice * b1_3 - b2_3;
            b2_3 = b1_3;
            b1_3 = b0;
            let b0 = coef + x4_twice * b1_4 - b2_4;
            b2_4 = b1_4;
            b1_4 = b0;
            let b0 = coef + x5_twice * b1_5 - b2_5;
            b2_5 = b1_5;
            b1_5 = b0;
            let b0 = coef + x6_twice * b1_6 - b2_6;
            b2_6 = b1_6;
            b1_6 = b0;
            let b0 = coef + x7_twice * b1_7 - b2_7;
            b2_7 = b1_7;
            b1_7 = b0;
            let b0 = coef + x8_twice * b1_8 - b2_8;
            b2_8 = b1_8;
            b1_8 = b0;
            let b0 = coef + x9_twice * b1_9 - b2_9;
            b2_9 = b1_9;
            b1_9 = b0;
            let b0 = coef + x10_twice * b1_10 - b2_10;
            b2_10 = b1_10;
            b1_10 = b0;
            let b0 = coef + x11_twice * b1_11 - b2_11;
            b2_11 = b1_11;
            b1_11 = b0;
            let b0 = coef + x12_twice * b1_12 - b2_12;
            b2_12 = b1_12;
            b1_12 = b0;
            let b0 = coef + x13_twice * b1_13 - b2_13;
            b2_13 = b1_13;
            b1_13 = b0;
            let b0 = coef + x14_twice * b1_14 - b2_14;
            b2_14 = b1_14;
            b1_14 = b0;
            let b0 = coef + x15_twice * b1_15 - b2_15;
            b2_15 = b1_15;
            b1_15 = b0;
            let b0 = coef + x16_twice * b1_16 - b2_16;
            b2_16 = b1_16;
            b1_16 = b0;
            let b0 = coef + x17_twice * b1_17 - b2_17;
            b2_17 = b1_17;
            b1_17 = b0;
            let b0 = coef + x18_twice * b1_18 - b2_18;
            b2_18 = b1_18;
            b1_18 = b0;
            let b0 = coef + x19_twice * b1_19 - b2_19;
            b2_19 = b1_19;
            b1_19 = b0;
            let b0 = coef + x20_twice * b1_20 - b2_20;
            b2_20 = b1_20;
            b1_20 = b0;
            let b0 = coef + x21_twice * b1_21 - b2_21;
            b2_21 = b1_21;
            b1_21 = b0;
            let b0 = coef + x22_twice * b1_22 - b2_22;
            b2_22 = b1_22;
            b1_22 = b0;
            let b0 = coef + x23_twice * b1_23 - b2_23;
            b2_23 = b1_23;
            b1_23 = b0;
            let b0 = coef + x24_twice * b1_24 - b2_24;
            b2_24 = b1_24;
            b1_24 = b0;
            let b0 = coef + x25_twice * b1_25 - b2_25;
            b2_25 = b1_25;
            b1_25 = b0;
            let b0 = coef + x26_twice * b1_26 - b2_26;
            b2_26 = b1_26;
            b1_26 = b0;
            let b0 = coef + x27_twice * b1_27 - b2_27;
            b2_27 = b1_27;
            b1_27 = b0;
            let b0 = coef + x28_twice * b1_28 - b2_28;
            b2_28 = b1_28;
            b1_28 = b0;
            let b0 = coef + x29_twice * b1_29 - b2_29;
            b2_29 = b1_29;
            b1_29 = b0;
            let b0 = coef + x30_twice * b1_30 - b2_30;
            b2_30 = b1_30;
            b1_30 = b0;
            let b0 = coef + x31_twice * b1_31 - b2_31;
            b2_31 = b1_31;
            b1_31 = b0;
        }};
    }

    let mut k = n;
    while k > 7 {
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
        k -= 1;
        step!(coefficients[k]);
    }
    while k > 0 {
        k -= 1;
        step!(coefficients[k]);
    }

    [
        b1_0 - x0 * b2_0,
        b1_1 - x1 * b2_1,
        b1_2 - x2 * b2_2,
        b1_3 - x3 * b2_3,
        b1_4 - x4 * b2_4,
        b1_5 - x5 * b2_5,
        b1_6 - x6 * b2_6,
        b1_7 - x7 * b2_7,
        b1_8 - x8 * b2_8,
        b1_9 - x9 * b2_9,
        b1_10 - x10 * b2_10,
        b1_11 - x11 * b2_11,
        b1_12 - x12 * b2_12,
        b1_13 - x13 * b2_13,
        b1_14 - x14 * b2_14,
        b1_15 - x15 * b2_15,
        b1_16 - x16 * b2_16,
        b1_17 - x17 * b2_17,
        b1_18 - x18 * b2_18,
        b1_19 - x19 * b2_19,
        b1_20 - x20 * b2_20,
        b1_21 - x21 * b2_21,
        b1_22 - x22 * b2_22,
        b1_23 - x23 * b2_23,
        b1_24 - x24 * b2_24,
        b1_25 - x25 * b2_25,
        b1_26 - x26 * b2_26,
        b1_27 - x27 * b2_27,
        b1_28 - x28 * b2_28,
        b1_29 - x29 * b2_29,
        b1_30 - x30 * b2_30,
        b1_31 - x31 * b2_31,
    ]
}

fn chebyshev_evaluate_direct(coefficients: &[f64], x_std: f64) -> f64 {
    let mut result = 0.0;
    for (k, &coef) in coefficients.iter().enumerate() {
        result += coef * chebyshev_polynomial(k, x_std);
    }
    result
}

pub struct ChebyshevCore {
    x_min: f64,
    x_max: f64,
    nodes: Vec<f64>,
    coefficients: Vec<f64>,
    n_points: usize,
    use_clenshaw: bool,
    fitted: bool,
}

impl ChebyshevCore {
    pub(crate) fn new(
        n_points: usize,
        x_min: f64,
        x_max: f64,
        use_clenshaw: bool,
    ) -> Result<Self, String> {
        if n_points == 0 {
            return Err("n_points must be positive".into());
        }
        if x_min >= x_max {
            return Err("x_min must be less than x_max".into());
        }
        let nodes = chebyshev_nodes(n_points, x_min, x_max);
        Ok(Self {
            x_min,
            x_max,
            nodes,
            coefficients: Vec::new(),
            n_points,
            use_clenshaw,
            fitted: false,
        })
    }

    pub(crate) fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    pub(crate) fn n_points(&self) -> usize {
        self.n_points
    }

    pub(crate) fn fit(&mut self, y: &[f64]) -> Result<(), String> {
        if y.len() != self.n_points {
            return Err(format!(
                "Expected {} y values (one for each Chebyshev node), got {}",
                self.n_points,
                y.len()
            ));
        }
        self.coefficients = compute_chebyshev_coefficients(y);
        self.fitted = true;
        Ok(())
    }

    pub(crate) fn coefficients(&self) -> Result<&[f64], String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(y) first.",
            }
            .into());
        }
        Ok(&self.coefficients)
    }

    pub(crate) fn set_method(&mut self, use_clenshaw: bool) {
        self.use_clenshaw = use_clenshaw;
    }

    fn ensure_in_range(&self, x: f64) -> Result<f64, String> {
        if x < self.x_min || x > self.x_max {
            return Err(format!(
                "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                x, self.x_min, self.x_max
            ));
        }
        Ok(transform_to_standard(x, self.x_min, self.x_max))
    }

    fn evaluate_impl(&self, x_std: f64) -> f64 {
        if self.use_clenshaw {
            chebyshev_evaluate_clenshaw(&self.coefficients, x_std)
        } else {
            chebyshev_evaluate_direct(&self.coefficients, x_std)
        }
    }

    pub(crate) fn repr(&self) -> String {
        let method = if self.use_clenshaw {
            "Clenshaw"
        } else {
            "Direct"
        };
        if self.fitted {
            format!(
                "ChebyshevInterpolator(n_points={}, x_range=[{:.2}, {:.2}], method={}, fitted)",
                self.n_points, self.x_min, self.x_max, method
            )
        } else {
            format!(
                "ChebyshevInterpolator(n_points={}, x_range=[{:.2}, {:.2}], method={}, not fitted)",
                self.n_points, self.x_min, self.x_max, method
            )
        }
    }
}

impl InterpolationCore for ChebyshevCore {
    fn evaluate_single(&self, x: f64) -> Result<f64, String> {
        if !self.fitted {
            return Err(CoreError::NotFitted {
                hint: "Call fit(y) first.",
            }
            .into());
        }
        let x_std = self.ensure_in_range(x)?;
        Ok(self.evaluate_impl(x_std))
    }

    fn fill_many(&self, xs: &[f64], out: &mut [f64]) -> Result<(), String> {
        if xs.len() != out.len() {
            return Err(CoreError::LengthMismatch {
                left_name: "input",
                left: xs.len(),
                right_name: "output",
                right: out.len(),
            }
            .into());
        }
        if self.use_clenshaw {
            let scale = 2.0 / (self.x_max - self.x_min);
            let offset = -1.0 - self.x_min * scale;
            let mut i = 0;

            macro_rules! checked_value {
                ($offset:expr) => {{
                    let value = xs[i + $offset];
                    if value < self.x_min || value > self.x_max {
                        return Err(format!(
                            "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                            value, self.x_min, self.x_max
                        ));
                    }
                    value
                }};
            }

            if self.x_min == -1.0 && self.x_max == 1.0 {
                while i + 31 < xs.len() {
                    let v0 = checked_value!(0);
                    let v1 = checked_value!(1);
                    let v2 = checked_value!(2);
                    let v3 = checked_value!(3);
                    let v4 = checked_value!(4);
                    let v5 = checked_value!(5);
                    let v6 = checked_value!(6);
                    let v7 = checked_value!(7);
                    let v8 = checked_value!(8);
                    let v9 = checked_value!(9);
                    let v10 = checked_value!(10);
                    let v11 = checked_value!(11);
                    let v12 = checked_value!(12);
                    let v13 = checked_value!(13);
                    let v14 = checked_value!(14);
                    let v15 = checked_value!(15);
                    let v16 = checked_value!(16);
                    let v17 = checked_value!(17);
                    let v18 = checked_value!(18);
                    let v19 = checked_value!(19);
                    let v20 = checked_value!(20);
                    let v21 = checked_value!(21);
                    let v22 = checked_value!(22);
                    let v23 = checked_value!(23);
                    let v24 = checked_value!(24);
                    let v25 = checked_value!(25);
                    let v26 = checked_value!(26);
                    let v27 = checked_value!(27);
                    let v28 = checked_value!(28);
                    let v29 = checked_value!(29);
                    let v30 = checked_value!(30);
                    let v31 = checked_value!(31);

                    let values = chebyshev_evaluate_clenshaw32(
                        &self.coefficients,
                        v0,
                        v1,
                        v2,
                        v3,
                        v4,
                        v5,
                        v6,
                        v7,
                        v8,
                        v9,
                        v10,
                        v11,
                        v12,
                        v13,
                        v14,
                        v15,
                        v16,
                        v17,
                        v18,
                        v19,
                        v20,
                        v21,
                        v22,
                        v23,
                        v24,
                        v25,
                        v26,
                        v27,
                        v28,
                        v29,
                        v30,
                        v31,
                    );
                    out[i] = values[0];
                    out[i + 1] = values[1];
                    out[i + 2] = values[2];
                    out[i + 3] = values[3];
                    out[i + 4] = values[4];
                    out[i + 5] = values[5];
                    out[i + 6] = values[6];
                    out[i + 7] = values[7];
                    out[i + 8] = values[8];
                    out[i + 9] = values[9];
                    out[i + 10] = values[10];
                    out[i + 11] = values[11];
                    out[i + 12] = values[12];
                    out[i + 13] = values[13];
                    out[i + 14] = values[14];
                    out[i + 15] = values[15];
                    out[i + 16] = values[16];
                    out[i + 17] = values[17];
                    out[i + 18] = values[18];
                    out[i + 19] = values[19];
                    out[i + 20] = values[20];
                    out[i + 21] = values[21];
                    out[i + 22] = values[22];
                    out[i + 23] = values[23];
                    out[i + 24] = values[24];
                    out[i + 25] = values[25];
                    out[i + 26] = values[26];
                    out[i + 27] = values[27];
                    out[i + 28] = values[28];
                    out[i + 29] = values[29];
                    out[i + 30] = values[30];
                    out[i + 31] = values[31];
                    i += 32;
                }
            }

            while i + 31 < xs.len() {
                let v0 = checked_value!(0);
                let v1 = checked_value!(1);
                let v2 = checked_value!(2);
                let v3 = checked_value!(3);
                let v4 = checked_value!(4);
                let v5 = checked_value!(5);
                let v6 = checked_value!(6);
                let v7 = checked_value!(7);
                let v8 = checked_value!(8);
                let v9 = checked_value!(9);
                let v10 = checked_value!(10);
                let v11 = checked_value!(11);
                let v12 = checked_value!(12);
                let v13 = checked_value!(13);
                let v14 = checked_value!(14);
                let v15 = checked_value!(15);
                let v16 = checked_value!(16);
                let v17 = checked_value!(17);
                let v18 = checked_value!(18);
                let v19 = checked_value!(19);
                let v20 = checked_value!(20);
                let v21 = checked_value!(21);
                let v22 = checked_value!(22);
                let v23 = checked_value!(23);
                let v24 = checked_value!(24);
                let v25 = checked_value!(25);
                let v26 = checked_value!(26);
                let v27 = checked_value!(27);
                let v28 = checked_value!(28);
                let v29 = checked_value!(29);
                let v30 = checked_value!(30);
                let v31 = checked_value!(31);

                let values = chebyshev_evaluate_clenshaw32(
                    &self.coefficients,
                    v0 * scale + offset,
                    v1 * scale + offset,
                    v2 * scale + offset,
                    v3 * scale + offset,
                    v4 * scale + offset,
                    v5 * scale + offset,
                    v6 * scale + offset,
                    v7 * scale + offset,
                    v8 * scale + offset,
                    v9 * scale + offset,
                    v10 * scale + offset,
                    v11 * scale + offset,
                    v12 * scale + offset,
                    v13 * scale + offset,
                    v14 * scale + offset,
                    v15 * scale + offset,
                    v16 * scale + offset,
                    v17 * scale + offset,
                    v18 * scale + offset,
                    v19 * scale + offset,
                    v20 * scale + offset,
                    v21 * scale + offset,
                    v22 * scale + offset,
                    v23 * scale + offset,
                    v24 * scale + offset,
                    v25 * scale + offset,
                    v26 * scale + offset,
                    v27 * scale + offset,
                    v28 * scale + offset,
                    v29 * scale + offset,
                    v30 * scale + offset,
                    v31 * scale + offset,
                );
                out[i] = values[0];
                out[i + 1] = values[1];
                out[i + 2] = values[2];
                out[i + 3] = values[3];
                out[i + 4] = values[4];
                out[i + 5] = values[5];
                out[i + 6] = values[6];
                out[i + 7] = values[7];
                out[i + 8] = values[8];
                out[i + 9] = values[9];
                out[i + 10] = values[10];
                out[i + 11] = values[11];
                out[i + 12] = values[12];
                out[i + 13] = values[13];
                out[i + 14] = values[14];
                out[i + 15] = values[15];
                out[i + 16] = values[16];
                out[i + 17] = values[17];
                out[i + 18] = values[18];
                out[i + 19] = values[19];
                out[i + 20] = values[20];
                out[i + 21] = values[21];
                out[i + 22] = values[22];
                out[i + 23] = values[23];
                out[i + 24] = values[24];
                out[i + 25] = values[25];
                out[i + 26] = values[26];
                out[i + 27] = values[27];
                out[i + 28] = values[28];
                out[i + 29] = values[29];
                out[i + 30] = values[30];
                out[i + 31] = values[31];
                i += 32;
            }

            while i + 15 < xs.len() {
                let v0 = checked_value!(0);
                let v1 = checked_value!(1);
                let v2 = checked_value!(2);
                let v3 = checked_value!(3);
                let v4 = checked_value!(4);
                let v5 = checked_value!(5);
                let v6 = checked_value!(6);
                let v7 = checked_value!(7);
                let v8 = checked_value!(8);
                let v9 = checked_value!(9);
                let v10 = checked_value!(10);
                let v11 = checked_value!(11);
                let v12 = checked_value!(12);
                let v13 = checked_value!(13);
                let v14 = checked_value!(14);
                let v15 = checked_value!(15);

                let values = chebyshev_evaluate_clenshaw16(
                    &self.coefficients,
                    v0 * scale + offset,
                    v1 * scale + offset,
                    v2 * scale + offset,
                    v3 * scale + offset,
                    v4 * scale + offset,
                    v5 * scale + offset,
                    v6 * scale + offset,
                    v7 * scale + offset,
                    v8 * scale + offset,
                    v9 * scale + offset,
                    v10 * scale + offset,
                    v11 * scale + offset,
                    v12 * scale + offset,
                    v13 * scale + offset,
                    v14 * scale + offset,
                    v15 * scale + offset,
                );
                out[i] = values[0];
                out[i + 1] = values[1];
                out[i + 2] = values[2];
                out[i + 3] = values[3];
                out[i + 4] = values[4];
                out[i + 5] = values[5];
                out[i + 6] = values[6];
                out[i + 7] = values[7];
                out[i + 8] = values[8];
                out[i + 9] = values[9];
                out[i + 10] = values[10];
                out[i + 11] = values[11];
                out[i + 12] = values[12];
                out[i + 13] = values[13];
                out[i + 14] = values[14];
                out[i + 15] = values[15];
                i += 16;
            }

            while i + 7 < xs.len() {
                let v0 = xs[i];
                if v0 < self.x_min || v0 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v0, self.x_min, self.x_max
                    ));
                }
                let v1 = xs[i + 1];
                if v1 < self.x_min || v1 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v1, self.x_min, self.x_max
                    ));
                }
                let v2 = xs[i + 2];
                if v2 < self.x_min || v2 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v2, self.x_min, self.x_max
                    ));
                }
                let v3 = xs[i + 3];
                if v3 < self.x_min || v3 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v3, self.x_min, self.x_max
                    ));
                }
                let v4 = xs[i + 4];
                if v4 < self.x_min || v4 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v4, self.x_min, self.x_max
                    ));
                }
                let v5 = xs[i + 5];
                if v5 < self.x_min || v5 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v5, self.x_min, self.x_max
                    ));
                }
                let v6 = xs[i + 6];
                if v6 < self.x_min || v6 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v6, self.x_min, self.x_max
                    ));
                }
                let v7 = xs[i + 7];
                if v7 < self.x_min || v7 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v7, self.x_min, self.x_max
                    ));
                }

                let values = chebyshev_evaluate_clenshaw8(
                    &self.coefficients,
                    v0 * scale + offset,
                    v1 * scale + offset,
                    v2 * scale + offset,
                    v3 * scale + offset,
                    v4 * scale + offset,
                    v5 * scale + offset,
                    v6 * scale + offset,
                    v7 * scale + offset,
                );
                out[i] = values[0];
                out[i + 1] = values[1];
                out[i + 2] = values[2];
                out[i + 3] = values[3];
                out[i + 4] = values[4];
                out[i + 5] = values[5];
                out[i + 6] = values[6];
                out[i + 7] = values[7];
                i += 8;
            }
            while i + 3 < xs.len() {
                let v0 = xs[i];
                if v0 < self.x_min || v0 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v0, self.x_min, self.x_max
                    ));
                }
                let v1 = xs[i + 1];
                if v1 < self.x_min || v1 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v1, self.x_min, self.x_max
                    ));
                }
                let v2 = xs[i + 2];
                if v2 < self.x_min || v2 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v2, self.x_min, self.x_max
                    ));
                }
                let v3 = xs[i + 3];
                if v3 < self.x_min || v3 > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        v3, self.x_min, self.x_max
                    ));
                }

                let values = chebyshev_evaluate_clenshaw4(
                    &self.coefficients,
                    v0 * scale + offset,
                    v1 * scale + offset,
                    v2 * scale + offset,
                    v3 * scale + offset,
                );
                out[i] = values[0];
                out[i + 1] = values[1];
                out[i + 2] = values[2];
                out[i + 3] = values[3];
                i += 4;
            }
            while i < xs.len() {
                let value = xs[i];
                if value < self.x_min || value > self.x_max {
                    return Err(format!(
                        "x value {:.4} is outside interpolation range [{:.4}, {:.4}]",
                        value, self.x_min, self.x_max
                    ));
                }
                out[i] = chebyshev_evaluate_clenshaw(&self.coefficients, value * scale + offset);
                i += 1;
            }
            return Ok(());
        }

        for (i, &value) in xs.iter().enumerate() {
            let x_std = self.ensure_in_range(value)?;
            out[i] = chebyshev_evaluate_direct(&self.coefficients, x_std);
        }
        Ok(())
    }
}
