/// Internal pseudo-random number generator using Linear Congruential Generator.
///
/// LCG: x_{n+1} = (a * x_n + c) mod m
/// Using Numerical Recipes parameters for decent quality.
///
/// A simple LCG-based pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    /// Create a new LCG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Generate the next random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = Self::A.wrapping_mul(self.state).wrapping_add(Self::C);
        self.state
    }

    /// Generate a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a random f64 in [a, b).
    pub fn next_range(&mut self, a: f64, b: f64) -> f64 {
        a + self.next_f64() * (b - a)
    }

    /// Generate a random usize in [0, n).
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcg_reproducible() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_lcg_different_seeds() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(43);
        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_lcg_f64_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!(x >= 0.0 && x < 1.0);
        }
    }

    #[test]
    fn test_lcg_custom_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..1000 {
            let x = rng.next_range(-5.0, 5.0);
            assert!(x >= -5.0 && x < 5.0);
        }
    }
}
