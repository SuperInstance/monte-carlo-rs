//! Uniform random sampling utilities.
//!
//! Generate uniform random samples in various spaces for Monte Carlo methods.

use crate::rng::Lcg;

/// Generate `n` uniform random samples in [a, b].
///
/// Returns a vector of `n` samples and the RNG state after generation.
pub fn uniform_samples(a: f64, b: f64, n: usize, seed: u64) -> (Vec<f64>, Lcg) {
    let mut rng = Lcg::new(seed);
    let samples: Vec<f64> = (0..n).map(|_| rng.next_range(a, b)).collect();
    (samples, rng)
}

/// Generate uniform random samples in a 2D rectangle [ax, bx] × [ay, by].
pub fn uniform_samples_2d(ax: f64, bx: f64, ay: f64, by: f64, n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| {
        let x = rng.next_range(ax, bx);
        let y = rng.next_range(ay, by);
        (x, y)
    }).collect()
}

/// Generate uniform samples on a d-dimensional hypercube [0, 1]^d.
pub fn uniform_samples_nd(d: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| {
        (0..d).map(|_| rng.next_f64()).collect()
    }).collect()
}

/// Compute the sample mean of a slice.
pub fn sample_mean(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty(), "Cannot compute mean of empty slice");
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Compute the sample variance (unbiased) of a slice.
pub fn sample_variance(samples: &[f64]) -> f64 {
    assert!(samples.len() > 1, "Need at least 2 samples for variance");
    let mean = sample_mean(samples);
    let n = samples.len() as f64;
    samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Compute the standard error of the mean.
pub fn standard_error(samples: &[f64]) -> f64 {
    (sample_variance(samples) / samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_samples_range() {
        let (samples, _) = uniform_samples(0.0, 1.0, 1000, 42);
        for s in &samples {
            assert!(*s >= 0.0 && *s < 1.0);
        }
    }

    #[test]
    fn test_uniform_samples_count() {
        let (samples, _) = uniform_samples(-1.0, 1.0, 500, 42);
        assert_eq!(samples.len(), 500);
    }

    #[test]
    fn test_uniform_2d_count() {
        let samples = uniform_samples_2d(0.0, 1.0, 0.0, 1.0, 100, 42);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_uniform_2d_range() {
        let samples = uniform_samples_2d(-2.0, 2.0, -3.0, 3.0, 500, 42);
        for (x, y) in &samples {
            assert!(*x >= -2.0 && *x < 2.0);
            assert!(*y >= -3.0 && *y < 3.0);
        }
    }

    #[test]
    fn test_sample_mean() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sample_mean(&samples) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_variance() {
        let samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // Known variance = 4.5714...
        let var = sample_variance(&samples);
        assert!((var - 4.571428).abs() < 0.01);
    }

    #[test]
    fn test_standard_error_decreases() {
        let (s100, _) = uniform_samples(0.0, 1.0, 100, 42);
        let (s10000, _) = uniform_samples(0.0, 1.0, 10000, 42);
        let se100 = standard_error(&s100);
        let se10000 = standard_error(&s10000);
        assert!(se10000 < se100);
    }
}
