//! # monte-carlo-rs Tutorial
//!
//! A progressive guide to Monte Carlo methods in Rust.
//! Run with: `cargo run --example tutorial`
//!
//! ## Lessons
//! 1. Random sampling — the foundation of Monte Carlo
//! 2. Monte Carlo integration — estimating integrals with randomness
//! 3. Estimating π — a classic Monte Carlo experiment
//! 4. Rejection sampling — sampling from arbitrary distributions
//! 5. Importance sampling — focus on what matters
//! 6. Antithetic variates — variance reduction via negative correlation
//! 7. Control variates & stratified sampling — advanced variance reduction
//! 8. Multi-dimensional integration — scaling to higher dimensions

use monte_carlo_rs::importance;
use monte_carlo_rs::integration;
use monte_carlo_rs::rejection;
use monte_carlo_rs::uniform;
use monte_carlo_rs::variance;

fn separator(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Lesson: {}", title);
    println!("═══════════════════════════════════════════════════════════");
    println!();
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         monte-carlo-rs Tutorial                          ║");
    println!("║   Monte Carlo Methods in Pure Rust                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    lesson_1_random_sampling();
    lesson_2_mc_integration();
    lesson_3_estimating_pi();
    lesson_4_rejection_sampling();
    lesson_5_importance_sampling();
    lesson_6_antithetic_variates();
    lesson_7_control_and_stratified();
    lesson_8_multidimensional();

    println!();
    println!("✅ Tutorial complete! You've mastered Monte Carlo methods in Rust.");
}

// ── Lesson 1 ────────────────────────────────────────────────────────
fn lesson_1_random_sampling() {
    separator("1. Random Sampling — The Foundation");

    println!("Every Monte Carlo method starts with random sampling.");
    println!("This crate provides a deterministic LCG (seeded RNG) for\n");
    println!("reproducible results with zero dependencies.\n");

    // 1D uniform samples
    let (samples, _) = uniform::uniform_samples(0.0, 1.0, 5, 42);
    println!("🎲 5 uniform samples in [0, 1):");
    println!("   {:?}", samples);

    // 2D samples
    let samples_2d = uniform::uniform_samples_2d(0.0, 1.0, 0.0, 1.0, 3, 42);
    println!("\n🎲 3 uniform samples in [0,1) × [0,1):");
    for (i, (x, y)) in samples_2d.iter().enumerate() {
        println!("   Point {}: ({:.6}, {:.6})", i + 1, x, y);
    }

    // Statistics
    let (many, _) = uniform::uniform_samples(0.0, 1.0, 10000, 42);
    let mean = uniform::sample_mean(&many);
    let var = uniform::sample_variance(&many);
    let se = uniform::standard_error(&many);
    println!("\n📊 Statistics for 10,000 samples in [0, 1):");
    println!("   Mean:            {:.6} (expected ≈ 0.5)", mean);
    println!("   Variance:        {:.6} (expected ≈ 1/12 = 0.0833)", var);
    println!("   Standard error:  {:.6}", se);
}

// ── Lesson 2 ────────────────────────────────────────────────────────
fn lesson_2_mc_integration() {
    separator("2. Monte Carlo Integration — Estimating Integrals");

    println!("Monte Carlo integration estimates ∫f(x)dx by averaging");
    println!("function evaluations at random points.\n");

    // ∫₀¹ x² dx = 1/3
    let n = 100_000;
    let result = integration::mc_integrate(|x| x * x, 0.0, 1.0, n, 42);
    let exact = 1.0 / 3.0;
    println!("📐 ∫₀¹ x² dx:");
    println!("   MC estimate:  {:.6}", result);
    println!("   Exact answer: {:.6}", exact);
    println!("   Error:        {:.6}", (result - exact).abs());

    // With error estimate
    let (est, se) = integration::mc_integrate_with_error(|x| x.sin(), 0.0, std::f64::consts::PI, 50000, 42);
    println!("\n📐 ∫₀^π sin(x) dx:");
    println!("   MC estimate:    {:.6}", est);
    println!("   Standard error: {:.6}", se);
    println!("   Exact answer:   2.000000");
    println!("   Within 2 SE?    {}", (est - 2.0).abs() < 2.0 * se);

    // Negative range
    let result_neg = integration::mc_integrate(|x| x * x, -1.0, 1.0, 100000, 42);
    println!("\n📐 ∫₋₁¹ x² dx:");
    println!("   MC estimate:  {:.6}", result_neg);
    println!("   Exact answer: {:.6}", 2.0 / 3.0);
}

// ── Lesson 3 ────────────────────────────────────────────────────────
fn lesson_3_estimating_pi() {
    separator("3. Estimating π — A Classic Monte Carlo Experiment");

    println!("Throw random darts at a unit square. The fraction that land");
    println!("inside the quarter circle estimates π/4.\n");

    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    println!("🎯 π estimates with increasing samples:\n");
    println!("   {:>12} {:>14} {:>14}", "Samples", "Estimate", "Error");
    println!("   {}", "-".repeat(44));
    for n in &sizes {
        let est = integration::estimate_pi(*n, 42);
        let err = (est - std::f64::consts::PI).abs();
        println!("   {:>12} {:>14.8} {:>14.8}", n, est, err);
    }
    println!("\n   → Error shrinks as ~1/√N (the Monte Carlo convergence rate)");
}

// ── Lesson 4 ────────────────────────────────────────────────────────
fn lesson_4_rejection_sampling() {
    separator("4. Rejection Sampling — Sampling Arbitrary Distributions");

    println!("Rejection sampling generates samples from any distribution");
    println!("by accepting/rejecting proposals under an envelope.\n");

    // Sample from triangular: f(x) = 2x on [0,1]
    let (samples, rate) = rejection::rejection_sample(
        |x| 2.0 * x,         // target: triangular
        |rng| rng.next_f64(), // proposal: uniform on [0,1)
        |_| 1.0,             // proposal PDF
        2.0,                  // envelope constant M
        10000,
        42,
    );
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("🔺 Triangular distribution f(x) = 2x:");
    println!("   Acceptance rate: {:.2}%", rate * 100.0);
    println!("   Sample mean:     {:.4} (expected 2/3 ≈ 0.6667)", mean);
    println!("   Sample count:    {}", samples.len());

    // Area under a curve = π/4
    let area = rejection::area_under_curve(
        |x| (1.0 - x * x).sqrt(),
        0.0, 1.0, 1.0,
        100000, 42,
    );
    println!("\n⭕ Area under √(1-x²) on [0,1] (= π/4):");
    println!("   MC estimate:  {:.6}", area);
    println!("   Exact (π/4):  {:.6}", std::f64::consts::PI / 4.0);

    // 2D rejection sampling
    let (pts_2d, rate_2d) = rejection::rejection_sample_2d(
        |x| x + 1.0, // f(x) = x + 1 on [0,1]
        0.0, 1.0, 2.0, // max value
        500, 42,
    );
    println!("\n 📊 2D rejection sampling under f(x) = x + 1:");
    println!("   Points:         {}", pts_2d.len());
    println!("   Acceptance rate: {:.2}%", rate_2d * 100.0);
}

// ── Lesson 5 ────────────────────────────────────────────────────────
fn lesson_5_importance_sampling() {
    separator("5. Importance Sampling — Focus on What Matters");

    println!("Importance sampling reduces variance by sampling more from");
    println!("regions that contribute most to the integral.\n");

    // Compare uniform vs importance sampling for ∫₀¹ x² dx
    let uniform_est = importance::importance_integrate(
        |x| x * x,
        |_| 1.0,              // uniform PDF
        |rng| rng.next_f64(), // uniform sample
        10000, 42,
    );
    let importance_est = importance::importance_integrate(
        |x| x * x,
        |x| 2.0 * x,                    // linear proposal: more samples near 1
        |rng| rng.next_f64().sqrt(),     // inverse CDF of 2x
        10000, 42,
    );
    println!("📐 ∫₀¹ x² dx with different proposals:");
    println!("   Uniform proposal:    {:.6} (error: {:.6})", uniform_est, (uniform_est - 1.0/3.0).abs());
    println!("   Linear proposal 2x:  {:.6} (error: {:.6})", importance_est, (importance_est - 1.0/3.0).abs());

    // Self-normalized importance sampling
    let (est, ess) = importance::self_normalized_is(
        |x| x,
        |_| 1.0,
        |rng| rng.next_f64(),
        10000, 42,
    );
    println!("\n📐 Self-normalized IS for E[x] on [0,1]:");
    println!("   Estimate:          {:.6} (expected 0.5)", est);
    println!("   Effective samples: {:.1} / 10000", ess);
    println!("   → ESS close to N means weights are uniform (good!)");
}

// ── Lesson 6 ────────────────────────────────────────────────────────
fn lesson_6_antithetic_variates() {
    separator("6. Antithetic Variates — Variance Reduction via Pairs");

    println!("Antithetic variates pair each sample u with (1-u), creating");
    println!("negatively correlated pairs that cancel variance.\n");

    // For linear functions, antithetic is EXACT
    let exact = variance::antithetic_integrate(|x| x, 0.0, 1.0, 1000, 42);
    println!("📐 ∫₀¹ x dx (antithetic = EXACT for linear functions!):");
    println!("   Result: {:.10} (exact: 0.5)", exact);

    // For non-linear, still reduces variance
    let (std_var, anti_var) = variance::antithetic_variance_comparison(
        |x| x * x, 0.0, 1.0, 10000, 42,
    );
    println!("\n📊 Variance comparison for ∫₀¹ x² dx:");
    println!("   Standard MC variance:  {:.6}", std_var);
    println!("   Antithetic variance:   {:.6}", anti_var);
    println!("   Reduction factor:      {:.2}×", std_var / anti_var);

    // Antithetic estimate
    let anti_est = variance::antithetic_integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 50000, 42);
    println!("\n📐 ∫₀^π sin(x) dx (antithetic):");
    println!("   Estimate: {:.6} (exact: 2.0)", anti_est);
}

// ── Lesson 7 ────────────────────────────────────────────────────────
fn lesson_7_control_and_stratified() {
    separator("7. Control Variates & Stratified Sampling");

    println!("Two more powerful variance reduction techniques.\n");

    // Control variate: ∫₀¹ x² dx using x as control (known mean = 0.5)
    let cv_est = variance::control_variate_integrate(
        |x| x * x,    // target
        |x| x,        // control (correlated, known mean)
        0.5,          // E[control]
        0.0, 1.0, 10000, 42,
    );
    println!("🎯 Control variate for ∫₀¹ x² dx:");
    println!("   Using x as control (E[x] = 0.5)");
    println!("   Estimate: {:.6} (exact: {:.6})", cv_est, 1.0/3.0);

    // Stratified sampling
    let strat_est = variance::stratified_integrate(
        |x| x.sin(), 0.0, std::f64::consts::PI,
        20,  // 20 strata
        500, // 500 samples per stratum
        42,
    );
    println!("\n🎯 Stratified sampling for ∫₀^π sin(x) dx:");
    println!("   20 strata × 500 samples = 10,000 total");
    println!("   Estimate: {:.6} (exact: 2.0)", strat_est);

    // Compare standard vs stratified
    let (std_est, strat_cmp) = variance::stratified_comparison(
        |x| x.sin(), 0.0, std::f64::consts::PI,
        10, 10000, 42,
    );
    println!("\n📊 Standard MC vs Stratified (same sample budget):");
    println!("   Standard:  {:.6}", std_est);
    println!("   Stratified: {:.6}", strat_cmp);
    println!("   Exact:      2.000000");
}

// ── Lesson 8 ────────────────────────────────────────────────────────
fn lesson_8_multidimensional() {
    separator("8. Multi-Dimensional Integration");

    println!("Monte Carlo shines in high dimensions where grid methods fail.\n");

    // ∫∫ 1 dx dy over [0,1]² = 1
    let vol_2d = integration::mc_integrate_nd(|_| 1.0, 2, 100000, 42);
    println!("📦 Volume of [0,1]² (2D):");
    println!("   MC estimate: {:.6} (exact: 1.0)", vol_2d);

    // ∫∫ x*y dx dy over [0,1]² = 1/4
    let int_2d = integration::mc_integrate_nd(|p| p[0] * p[1], 2, 100000, 42);
    println!("\n📐 ∫∫ x·y dx dy over [0,1]²:");
    println!("   MC estimate: {:.6} (exact: 0.25)", int_2d);

    // Volume of unit sphere in d dimensions
    println!("\n🔮 Volume of unit sphere in d dimensions:");
    println!("   (fraction of [0,1]^d where sum of squares ≤ 1)\n");
    println!("   {:>3} {:>12} {:>14} {:>14}", "Dim", "MC estimate", "Exact", "Error");
    println!("   {}", "-".repeat(46));

    let exact_volumes = [2.0, std::f64::consts::PI, 4.0/3.0*std::f64::consts::PI, std::f64::consts::PI*std::f64::consts::PI/2.0];
    // 2^d * fraction in positive quadrant
    let dims = [1, 2, 3, 4];
    let powers = [2.0, 4.0, 8.0, 16.0];
    for (i, &d) in dims.iter().enumerate() {
        let frac = integration::mc_integrate_nd(|p| {
            if p.iter().map(|x| x * x).sum::<f64>() <= 1.0 { 1.0 } else { 0.0 }
        }, d, 500000, 42);
        let vol = powers[i] * frac;
        let err = (vol - exact_volumes[i]).abs();
        println!("   {:>3} {:>12.6} {:>14.6} {:>14.6}", d, vol, exact_volumes[i], err);
    }

    // High-dimensional: curse of dimensionality
    println!("\n📉 Fraction of hypercube inside sphere (curse of dimensionality):");
    for d in [2, 5, 10, 20] {
        let frac = integration::mc_integrate_nd(|p| {
            if p.iter().map(|x| x * x).sum::<f64>() <= 1.0 { 1.0 } else { 0.0 }
        }, d, 200000, 42);
        println!("   d={:>2}: {:.6} ({:.2}% of hypercube)", d, frac, frac * 100.0);
    }
}
