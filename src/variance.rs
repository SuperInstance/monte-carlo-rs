//! Variance reduction techniques.
//!
//! Methods to reduce the variance of Monte Carlo estimates without
//! increasing the number of samples.

use crate::rng::Lcg;
use crate::uniform::{sample_mean, sample_variance};

/// Antithetic variates: use paired negatively correlated samples.
///
/// For each random u, also use (1-u). This reduces variance for
/// monotone functions.
pub fn antithetic_integrate<F>(f: F, a: f64, b: f64, n: usize, seed: u64) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let half_n = n / 2;
    let mut sum = 0.0;
    for _ in 0..half_n {
        let u = rng.next_f64();
        let x1 = a + u * (b - a);
        let x2 = a + (1.0 - u) * (b - a);
        sum += f(x1) + f(x2);
    }
    (b - a) * sum / (2 * half_n) as f64
}

/// Compare variance of standard MC vs antithetic variates.
///
/// Returns (standard_variance, antithetic_variance).
pub fn antithetic_variance_comparison<F>(f: F, a: f64, b: f64, n: usize, seed: u64) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Standard MC
    let mut rng = Lcg::new(seed);
    let std_samples: Vec<f64> = (0..n).map(|_| f(rng.next_range(a, b))).collect();
    let std_var = sample_variance(&std_samples);

    // Antithetic
    let mut rng2 = Lcg::new(seed);
    let half_n = n / 2;
    let anti_samples: Vec<f64> = (0..half_n).map(|_| {
        let u = rng2.next_f64();
        (f(a + u * (b - a)) + f(a + (1.0 - u) * (b - a))) / 2.0
    }).collect();
    let anti_var = sample_variance(&anti_samples);

    (std_var, anti_var)
}

/// Control variates: use a correlated control variable with known expectation.
///
/// If Y = f(X) and C = control(X) with E[C] known, then
/// Y_cv = Y - β*(C - E[C]) has lower variance.
pub fn control_variate_integrate<F, C>(
    f: F,
    control: C,
    control_mean: f64,
    a: f64,
    b: f64,
    n: usize,
    seed: u64,
) -> f64
where
    F: Fn(f64) -> f64,
    C: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let f_samples: Vec<f64> = (0..n).map(|_| f(rng.next_range(a, b))).collect();
    let mut rng2 = Lcg::new(seed);
    let c_samples: Vec<f64> = (0..n).map(|_| control(rng2.next_range(a, b))).collect();

    // Estimate optimal β via covariance/variance
    let f_mean = sample_mean(&f_samples);
    let c_mean = sample_mean(&c_samples);
    let c_var = sample_variance(&c_samples);

    let cov: f64 = f_samples.iter()
        .zip(c_samples.iter())
        .map(|(f, c)| (f - f_mean) * (c - c_mean))
        .sum::<f64>() / (n - 1) as f64;

    let beta = if c_var > 0.0 { cov / c_var } else { 0.0 };

    let adjusted: Vec<f64> = f_samples.iter()
        .zip(c_samples.iter())
        .map(|(f, c)| f - beta * (c - control_mean))
        .collect();

    (b - a) * sample_mean(&adjusted)
}

/// Stratified sampling: divide the integration domain into strata
/// and sample from each.
pub fn stratified_integrate<F>(f: F, a: f64, b: f64, n_strata: usize, samples_per_stratum: usize, seed: u64) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let stratum_width = (b - a) / n_strata as f64;
    let mut total = 0.0;
    let total_samples = n_strata * samples_per_stratum;

    for i in 0..n_strata {
        let stratum_a = a + i as f64 * stratum_width;
        let stratum_b = stratum_a + stratum_width;
        for _ in 0..samples_per_stratum {
            let x = rng.next_range(stratum_a, stratum_b);
            total += f(x);
        }
    }

    (b - a) * total / total_samples as f64
}

/// Compare standard MC vs stratified sampling variance.
///
/// Returns (standard_estimate, stratified_estimate) for the same total samples.
pub fn stratified_comparison<F>(f: F, a: f64, b: f64, n_strata: usize, total_samples: usize, seed: u64) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let std_est = crate::integration::mc_integrate(&f, a, b, total_samples, seed);
    let samples_per = total_samples / n_strata;
    let strat_est = stratified_integrate(&f, a, b, n_strata, samples_per.max(1), seed);
    (std_est, strat_est)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_antithetic_integrate_linear() {
        // ∫_0^1 x dx = 0.5 (antithetic should be exact for linear!)
        let result = antithetic_integrate(|x| x, 0.0, 1.0, 1000, 42);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_antithetic_integrate_quadratic() {
        // ∫_0^1 x² dx = 1/3
        let result = antithetic_integrate(|x| x * x, 0.0, 1.0, 10000, 42);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_antithetic_reduces_variance() {
        // For monotone functions, antithetic should reduce variance
        let (std_var, anti_var) = antithetic_variance_comparison(
            |x| x * x, 0.0, 1.0, 10000, 42,
        );
        assert!(anti_var < std_var);
    }

    #[test]
    fn test_control_variate_integrate() {
        // ∫_0^1 x² dx = 1/3, using x as control (known mean = 0.5)
        let result = control_variate_integrate(
            |x| x * x,
            |x| x,
            0.5,
            0.0, 1.0, 10000, 42,
        );
        assert!((result - 1.0 / 3.0).abs() < 0.02);
    }

    #[test]
    fn test_stratified_integrate() {
        let result = stratified_integrate(|x| x * x, 0.0, 1.0, 10, 1000, 42);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_stratified_integrate_sine() {
        let result = stratified_integrate(
            |x| x.sin(),
            0.0, std::f64::consts::PI, 20, 500, 42,
        );
        assert!((result - 2.0).abs() < 0.02);
    }

    #[test]
    fn test_stratified_comparison() {
        let (std_est, strat_est) = stratified_comparison(
            |x| x.sin(), 0.0, std::f64::consts::PI, 10, 10000, 42,
        );
        // Both should be close to 2.0
        assert!((std_est - 2.0).abs() < 0.05);
        assert!((strat_est - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_antithetic_integrate_constant() {
        let result = antithetic_integrate(|_| 3.0, 0.0, 1.0, 1000, 42);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stratified_constant() {
        let result = stratified_integrate(|_| 5.0, 0.0, 2.0, 5, 100, 42);
        assert!((result - 10.0).abs() < 1e-10);
    }
}
