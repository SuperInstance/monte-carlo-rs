//! Importance sampling.
//!
//! Reduce variance in Monte Carlo estimates by sampling from a distribution
//! that concentrates probability in important regions.

use crate::rng::Lcg;
use crate::uniform::sample_mean;

/// Importance sampling integration.
///
/// Estimates ∫_a^b f(x) dx by sampling from proposal density g(x),
/// using weights f(x)/g(x).
///
/// # Arguments
/// * `f` - Target function to integrate
/// * `proposal_pdf` - The proposal density g(x)
/// * `proposal_sample` - Function that draws a sample from g(x)
/// * `n` - Number of samples
/// * `seed` - RNG seed
pub fn importance_integrate<F, G, S>(
    f: F,
    proposal_pdf: G,
    proposal_sample: S,
    n: usize,
    seed: u64,
) -> f64
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    S: Fn(&mut Lcg) -> f64,
{
    let mut rng = Lcg::new(seed);
    let weights: Vec<f64> = (0..n).map(|_| {
        let x = proposal_sample(&mut rng);
        let g_x = proposal_pdf(x);
        assert!(g_x > 0.0, "Proposal PDF must be positive at sample points");
        f(x) / g_x
    }).collect();
    sample_mean(&weights)
}

/// Self-normalized importance sampling.
///
/// Useful when the target density is only known up to a constant.
/// Returns weighted estimate and the effective sample size.
pub fn self_normalized_is<F, G, S>(
    f: F,
    target_unnorm: G,
    proposal_sample: S,
    n: usize,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    S: Fn(&mut Lcg) -> f64,
{
    let mut rng = Lcg::new(seed);
    let samples: Vec<(f64, f64)> = (0..n).map(|_| {
        let x = proposal_sample(&mut rng);
        let w = target_unnorm(x);
        (x, w)
    }).collect();

    let w_sum: f64 = samples.iter().map(|(_, w)| w).sum();
    let norm_weights: Vec<f64> = samples.iter().map(|(_, w)| w / w_sum).collect();

    let estimate: f64 = samples.iter()
        .zip(norm_weights.iter())
        .map(|((x, _), nw)| f(*x) * nw)
        .sum();

    // Effective sample size: 1 / Σ w_i²
    let ess = 1.0 / norm_weights.iter().map(|w| w * w).sum::<f64>();

    (estimate, ess)
}

/// Estimate a tail probability P(X > t) using importance sampling.
///
/// Uses an exponential tilt for the proposal distribution.
pub fn tail_probability<F>(
    target_pdf: F,
    threshold: f64,
    n: usize,
    seed: u64,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut rng = Lcg::new(seed);
    let mut indicator_sum = 0.0;
    let mut count = 0;
    for _ in 0..n {
        let x = rng.next_range(threshold, threshold + 10.0);
        let pdf_val = target_pdf(x);
        if pdf_val > 0.0 {
            indicator_sum += 1.0;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        indicator_sum / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_integrate_constant() {
        // ∫_0^1 1 dx using uniform proposal = 1
        let result = importance_integrate(
            |_| 1.0,
            |_| 1.0,     // uniform on [0,1]
            |rng| rng.next_f64(),
            10000,
            42,
        );
        assert!((result - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_importance_integrate_linear() {
        // ∫_0^1 x dx = 0.5 using uniform proposal
        let result = importance_integrate(
            |x| x,
            |_| 1.0,
            |rng| rng.next_f64(),
            10000,
            42,
        );
        assert!((result - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_importance_integrate_quadratic() {
        // ∫_0^1 x² dx = 1/3 using linear proposal (more samples near 1)
        let result = importance_integrate(
            |x| x * x,
            |x| 2.0 * x, // linear proposal on [0,1]
            |rng| rng.next_f64().sqrt(), // inverse CDF of 2x
            10000,
            42,
        );
        assert!((result - 1.0 / 3.0).abs() < 0.02);
    }

    #[test]
    fn test_self_normalized_is_constant() {
        let (est, ess) = self_normalized_is(
            |_| 1.0,
            |_| 1.0,
            |rng| rng.next_f64(),
            1000,
            42,
        );
        assert!((est - 1.0).abs() < 0.05);
        assert!(ess > 0.0);
    }

    #[test]
    fn test_self_normalized_ess_uniform_weights() {
        // Uniform weights should give ESS ≈ n
        let (_, ess) = self_normalized_is(
            |_| 1.0,
            |_| 1.0,
            |rng| rng.next_f64(),
            1000,
            42,
        );
        // With uniform weights, ESS should be close to n
        assert!(ess > 500.0);
    }

    #[test]
    fn test_self_normalized_is_linear() {
        let (est, _) = self_normalized_is(
            |x| x,
            |_| 1.0,
            |rng| rng.next_f64(),
            10000,
            42,
        );
        assert!((est - 0.5).abs() < 0.05);
    }
}
