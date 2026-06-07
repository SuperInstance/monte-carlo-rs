//! Rejection sampling.
//!
//! Generate samples from a target distribution using a proposal distribution
//! and an envelope function.

use crate::rng::Lcg;

/// Rejection sampling to generate samples from a target distribution.
///
/// Given a target density f(x) ≤ M * g(x), where g(x) is the proposal:
/// 1. Sample x from g(x)
/// 2. Sample u ~ Uniform(0, 1)
/// 3. Accept x if u ≤ f(x) / (M * g(x))
///
/// # Arguments
/// * `target_pdf` - Target density f(x) (need not be normalized)
/// * `proposal_sample` - Function to sample from proposal g(x)
/// * `proposal_pdf` - Proposal density g(x)
/// * `m` - Envelope constant M such that f(x) ≤ M * g(x)
/// * `n` - Desired number of accepted samples
/// * `seed` - RNG seed
///
/// # Returns
/// Vector of accepted samples and the acceptance rate.
pub fn rejection_sample<F, G, S>(
    target_pdf: F,
    proposal_sample: S,
    proposal_pdf: G,
    m: f64,
    n: usize,
    seed: u64,
) -> (Vec<f64>, f64)
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    S: Fn(&mut Lcg) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut accepted = Vec::new();
    let mut total_attempts = 0usize;

    while accepted.len() < n {
        let x = proposal_sample(&mut rng);
        let u = rng.next_f64();
        let g_x = proposal_pdf(x);
        let f_x = target_pdf(x);
        if g_x > 0.0 && u <= f_x / (m * g_x) {
            accepted.push(x);
        }
        total_attempts += 1;
        // Safety limit
        if total_attempts > n * 100 {
            break;
        }
    }

    let acceptance_rate = accepted.len() as f64 / total_attempts as f64;
    (accepted, acceptance_rate)
}

/// Rejection sampling in 2D.
///
/// Sample (x, y) uniformly from the region under f(x) in [a, b].
pub fn rejection_sample_2d<F>(
    f: F,
    a: f64,
    b: f64,
    max_f: f64,
    n: usize,
    seed: u64,
) -> (Vec<(f64, f64)>, f64)
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut accepted = Vec::new();
    let mut total = 0usize;

    while accepted.len() < n {
        let x = rng.next_range(a, b);
        let y = rng.next_range(0.0, max_f);
        if y <= f(x) {
            accepted.push((x, y));
        }
        total += 1;
        if total > n * 100 {
            break;
        }
    }

    let rate = accepted.len() as f64 / total as f64;
    (accepted, rate)
}

/// Estimate the area under a curve using rejection sampling.
///
/// Samples uniformly in the bounding box [a, b] × [0, max_f]
/// and counts the fraction below the curve.
pub fn area_under_curve<F>(f: F, a: f64, b: f64, max_f: f64, n: usize, seed: u64) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut inside = 0usize;
    for _ in 0..n {
        let x = rng.next_range(a, b);
        let y = rng.next_range(0.0, max_f);
        if y <= f(x) {
            inside += 1;
        }
    }
    (b - a) * max_f * inside as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejection_sample_uniform() {
        // Sample from uniform on [0,1] using uniform proposal
        let (samples, rate) = rejection_sample(
            |_| 1.0,
            |rng| rng.next_f64(),
            |_| 1.0,
            1.0,
            1000,
            42,
        );
        assert_eq!(samples.len(), 1000);
        assert!((rate - 1.0).abs() < 0.01); // Should accept all
    }

    #[test]
    fn test_rejection_sample_triangle() {
        // Sample from triangular distribution f(x) = 2x on [0,1]
        // Using uniform proposal g(x) = 1, M = 2
        let (samples, _rate) = rejection_sample(
            |x| 2.0 * x,
            |rng| rng.next_f64(),
            |_| 1.0,
            2.0,
            5000,
            42,
        );
        // Mean of triangular(2x) on [0,1] = 2/3
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 2.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_rejection_acceptance_rate_bounds() {
        let (_, rate) = rejection_sample(
            |x| 2.0 * x,
            |rng| rng.next_f64(),
            |_| 1.0,
            2.0,
            1000,
            42,
        );
        // Acceptance rate should be between 0 and 1
        assert!(rate > 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_rejection_2d_count() {
        let (samples, _) = rejection_sample_2d(
            |x| x + 1.0, // line from (0,1) to (1,2)
            0.0,
            1.0,
            2.0,
            100,
            42,
        );
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_area_under_curve_constant() {
        // Area under f(x) = 1 on [0,1] = 1
        let area = area_under_curve(|_| 1.0, 0.0, 1.0, 1.0, 10000, 42);
        assert!((area - 1.0).abs() < 0.03);
    }

    #[test]
    fn test_area_under_curve_quadratic() {
        // Area under f(x) = x² on [0,1] = 1/3
        let area = area_under_curve(|x| x * x, 0.0, 1.0, 1.0, 50000, 42);
        assert!((area - 1.0 / 3.0).abs() < 0.02);
    }

    #[test]
    fn test_area_circle_quarter() {
        // Area of quarter circle = π/4
        let area = area_under_curve(
            |x| (1.0 - x * x).sqrt(),
            0.0, 1.0, 1.0, 50000, 42,
        );
        assert!((area - std::f64::consts::PI / 4.0).abs() < 0.03);
    }
}
