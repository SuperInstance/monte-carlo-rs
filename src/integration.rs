//! Monte Carlo integration.
//!
//! Estimate definite integrals using Monte Carlo methods.

use crate::rng::Lcg;
use crate::uniform::sample_mean;

/// Simple Monte Carlo integration of f(x) over [a, b].
///
/// Estimates ∫_a^b f(x) dx ≈ (b-a) * (1/N) Σ f(x_i)
///
/// # Arguments
/// * `f` - The function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of samples
/// * `seed` - RNG seed
pub fn mc_integrate<F>(f: F, a: f64, b: f64, n: usize, seed: u64) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let samples: Vec<f64> = (0..n).map(|_| f(rng.next_range(a, b))).collect();
    (b - a) * sample_mean(&samples)
}

/// Monte Carlo integration with confidence interval.
///
/// Returns (estimate, standard_error).
pub fn mc_integrate_with_error<F>(f: F, a: f64, b: f64, n: usize, seed: u64) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let samples: Vec<f64> = (0..n).map(|_| f(rng.next_range(a, b))).collect();
    let mean = sample_mean(&samples);
    let var = if samples.len() > 1 {
        let ssq: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        ssq / samples.len() as f64
    } else {
        0.0
    };
    ((b - a) * mean, (b - a) * var.sqrt())
}

/// Estimate π using Monte Carlo sampling in a unit square.
///
/// Randomly samples points in [0,1]² and counts those inside
/// the unit circle quarter. π ≈ 4 * (points inside) / (total points).
pub fn estimate_pi(n: usize, seed: u64) -> f64 {
    let mut rng = Lcg::new(seed);
    let mut inside = 0usize;
    for _ in 0..n {
        let x = rng.next_f64();
        let y = rng.next_f64();
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }
    4.0 * inside as f64 / n as f64
}

/// Multi-dimensional Monte Carlo integration.
///
/// Estimates ∫ f(x) dx over [0, 1]^d ≈ (1/N) Σ f(x_i)
pub fn mc_integrate_nd<F>(f: F, d: usize, n: usize, seed: u64) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut rng = Lcg::new(seed);
    let samples: Vec<f64> = (0..n).map(|_| {
        let point: Vec<f64> = (0..d).map(|_| rng.next_f64()).collect();
        f(&point)
    }).collect();
    sample_mean(&samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        // ∫_0^1 1 dx = 1
        let result = mc_integrate(|_| 1.0, 0.0, 1.0, 10000, 42);
        assert!((result - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_integrate_linear() {
        // ∫_0^1 x dx = 0.5
        let result = mc_integrate(|x| x, 0.0, 1.0, 10000, 42);
        assert!((result - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_integrate_quadratic() {
        // ∫_0^1 x² dx = 1/3
        let result = mc_integrate(|x| x * x, 0.0, 1.0, 10000, 42);
        assert!((result - 1.0 / 3.0).abs() < 0.02);
    }

    #[test]
    fn test_integrate_sine() {
        // ∫_0^π sin(x) dx = 2
        let result = mc_integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 50000, 42);
        assert!((result - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_integrate_negative_range() {
        // ∫_{-1}^1 x² dx = 2/3
        let result = mc_integrate(|x| x * x, -1.0, 1.0, 10000, 42);
        assert!((result - 2.0 / 3.0).abs() < 0.03);
    }

    #[test]
    fn test_estimate_pi() {
        let pi_est = estimate_pi(100000, 42);
        assert!((pi_est - std::f64::consts::PI).abs() < 0.05);
    }

    #[test]
    fn test_estimate_pi_accuracy_improves() {
        let pi1 = estimate_pi(1000, 42);
        let pi2 = estimate_pi(100000, 42);
        let err1 = (pi1 - std::f64::consts::PI).abs();
        let err2 = (pi2 - std::f64::consts::PI).abs();
        // More samples should generally give better accuracy
        assert!(err2 < err1 * 1.5); // Allow some stochastic slack
    }

    #[test]
    fn test_integrate_with_error() {
        let (est, se) = mc_integrate_with_error(|x| x, 0.0, 1.0, 10000, 42);
        assert!((est - 0.5).abs() < 0.02);
        assert!(se > 0.0);
    }

    #[test]
    fn test_integrate_nd() {
        // ∫∫ 1 dx dy over [0,1]² = 1
        let result = mc_integrate_nd(|_| 1.0, 2, 10000, 42);
        assert!((result - 1.0).abs() < 0.03);
    }
}
