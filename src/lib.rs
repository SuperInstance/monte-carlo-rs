//! # monte-carlo-rs
//!
//! A pure-Rust Monte Carlo methods library providing:
//! - Uniform random sampling
//! - Importance sampling
//! - Rejection sampling
//! - Monte Carlo integration
//! - Variance reduction techniques
//!
//! All random number generation uses a simple Linear Congruential Generator (LCG)
//! so no external dependencies are needed.
//!
//! ## Example
//!
//! ```
//! use monte_carlo_rs::integration::mc_integrate;
//!
//! // Estimate integral of f(x) = x^2 on [0, 1] = 1/3
//! let result = mc_integrate(|x| x * x, 0.0, 1.0, 10_000, 42);
//! assert!((result - 1.0/3.0).abs() < 0.05);
//! ```

/// Pseudo-random number generation utilities.
mod rng;

/// Uniform random sampling.
pub mod uniform;

/// Importance sampling.
pub mod importance;

/// Rejection sampling.
pub mod rejection;

/// Monte Carlo integration.
pub mod integration;

/// Variance reduction techniques.
pub mod variance;
