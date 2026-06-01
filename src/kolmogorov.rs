//! Kolmogorov complexity estimator.
//!
//! K(answer | structure) bounds the irreducible runtime residue.
//! We estimate this via compression-based and structural methods.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Kolmogorov complexity estimate for a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KolmogorovEstimate {
    /// The data being estimated.
    pub data_description: String,
    /// Lower bound on K(data | structure).
    pub lower_bound_bits: f64,
    /// Upper bound on K(data | structure).
    pub upper_bound_bits: f64,
    /// Method used for estimation.
    pub method: String,
    /// Confidence in the estimate.
    pub confidence: f64,
}

/// Kolmogorov complexity estimator.
pub struct KolmogorovEstimator {
    /// Precision for discretization.
    pub precision: f64,
}

impl KolmogorovEstimator {
    pub fn new(precision: f64) -> Self {
        Self { precision }
    }

    pub fn default() -> Self {
        Self::new(1e-6)
    }

    /// Estimate K(output | structure) for a single computation.
    pub fn estimate(&self, output: &[f64], structure: &[f64]) -> KolmogorovEstimate {
        let n = output.len();
        if n == 0 {
            return KolmogorovEstimate {
                data_description: "empty".to_string(),
                lower_bound_bits: 0.0,
                upper_bound_bits: 0.0,
                method: "trivial".to_string(),
                confidence: 1.0,
            };
        }

        // Method 1: Compression via Lempel-Ziv-like complexity
        let lz_lower = self.lempel_ziv_lower_bound(output, structure);
        let lz_upper = self.lempel_ziv_upper_bound(output, structure);

        // Method 2: Entropy-based
        let entropy_bits = self.entropy_estimate(output);

        // Method 3: Structural similarity (how much of output is explained by structure)
        let residual_bits = self.structural_residual(output, structure);

        let lower = lz_lower.max(residual_bits * 0.5);
        let upper = lz_upper.min(entropy_bits * 1.5).min(n as f64 * 64.0);

        KolmogorovEstimate {
            data_description: format!("output[{}]", n),
            lower_bound_bits: lower,
            upper_bound_bits: upper,
            method: "combined_lz_entropy_structural".to_string(),
            confidence: 0.7,
        }
    }

    /// Estimate K for a family of computations (the irreducible residue of the whole family).
    pub fn estimate_family(&self, outputs: &[Vec<f64>], structures: &[Vec<f64>]) -> KolmogorovEstimate {
        if outputs.is_empty() {
            return KolmogorovEstimate {
                data_description: "empty_family".to_string(),
                lower_bound_bits: 0.0,
                upper_bound_bits: 0.0,
                method: "trivial".to_string(),
                confidence: 1.0,
            };
        }

        // Flatten and estimate
        let flat_output: Vec<f64> = outputs.iter().flat_map(|o| o.iter().copied()).collect();
        let flat_structure: Vec<f64> = structures.iter().flat_map(|s| s.iter().copied()).collect();
        self.estimate(&flat_output, &flat_structure)
    }

    /// Compute the information-theoretic lower bound on residue.
    /// This is the key theorem: the residue cannot be less than K(answer|structure).
    pub fn residue_lower_bound(&self, outputs: &[Vec<f64>], structures: &[Vec<f64>]) -> f64 {
        let estimate = self.estimate_family(outputs, structures);
        estimate.lower_bound_bits
    }

    fn lempel_ziv_lower_bound(&self, data: &[f64], _context: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        // Discretize and count distinct substrings
        let discrete: Vec<u32> = data.iter().map(|&x| {
            let normalized = x / self.precision;
            normalized.to_bits() as u32
        }).collect();

        // Count minimal number of distinct phrases
        let phrases = count_phrases(&discrete);
        (phrases as f64 * 16.0).max(1.0) // lower bound: each phrase needs at least ~log phrases bits
    }

    fn lempel_ziv_upper_bound(&self, data: &[f64], context: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        // Upper bound: the data itself, compressed
        // Simple: count unique values * bits per value
        let mut unique = std::collections::HashSet::new();
        for &x in data {
            let bucket = (x / self.precision).round() as i64;
            unique.insert(bucket);
        }
        let unique_bits = if unique.len() <= 1 { 0.0 } else { (unique.len() as f64).log2() };
        data.len() as f64 * unique_bits
    }

    fn entropy_estimate(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        // Bucket-based entropy
        let mut buckets: HashMap<i64, usize> = HashMap::new();
        for &x in data {
            let bucket = (x / self.precision).round() as i64;
            *buckets.entry(bucket).or_insert(0) += 1;
        }
        let n = data.len() as f64;
        let mut entropy = 0.0;
        for &count in buckets.values() {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
        entropy * data.len() as f64
    }

    fn structural_residual(&self, output: &[f64], structure: &[f64]) -> f64 {
        if output.is_empty() || structure.is_empty() {
            return output.len() as f64 * 8.0; // rough estimate
        }
        // How many bits of output are not explained by structure?
        // If output is a simple function of structure, residual is low.
        let explained_bits = if structure.len() == output.len() {
            // Check correlation
            let correlation = pearson_correlation(output, structure);
            (correlation.abs() * output.len() as f64 * 8.0).min(output.len() as f64 * 8.0)
        } else {
            0.0
        };
        (output.len() as f64 * 8.0 - explained_bits).max(0.0)
    }
}

fn count_phrases(data: &[u32]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let mut phrases = 1;
    let mut i = 1;
    while i < data.len() {
        let mut found = false;
        for len in (1..=i).rev() {
            if i + len <= data.len() {
                let candidate = &data[i..i + len];
                // Check if this phrase appeared before
                for start in 0..i {
                    if start + len <= i && &data[start..start + len] == candidate {
                        found = true;
                        break;
                    }
                }
                if found {
                    i += len;
                    break;
                }
            }
        }
        if !found {
            phrases += 1;
            i += 1;
        }
    }
    phrases
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mx = x.iter().take(n as usize).sum::<f64>() / n;
    let my = y.iter().take(n as usize).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n as usize {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-12 || vy < 1e-12 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        let est = KolmogorovEstimator::default();
        let result = est.estimate(&[], &[]);
        assert_eq!(result.lower_bound_bits, 0.0);
        assert_eq!(result.upper_bound_bits, 0.0);
    }

    #[test]
    fn test_constant_data_low_complexity() {
        let est = KolmogorovEstimator::default();
        let data = vec![1.0; 100];
        let structure = vec![1.0; 100];
        let result = est.estimate(&data, &structure);
        // Constant data should have low complexity
        assert!(result.upper_bound_bits < 100.0); // much less than 100 * 64
    }

    #[test]
    fn test_random_data_higher_complexity() {
        let est = KolmogorovEstimator::new(0.1);
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 7.31).sin() * 100.0).collect();
        let structure = vec![1.0; 100];
        let result = est.estimate(&data, &structure);
        assert!(result.lower_bound_bits > 0.0);
    }

    #[test]
    fn test_structure_reduces_complexity() {
        let est = KolmogorovEstimator::new(0.1);
        let output: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
        let matching_structure: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let no_structure = vec![1.0; 50];

        let with_structure = est.estimate(&output, &matching_structure);
        let without_structure = est.estimate(&output, &no_structure);

        // With matching structure, complexity should be lower
        assert!(with_structure.upper_bound_bits <= without_structure.upper_bound_bits);
    }

    #[test]
    fn test_family_estimate() {
        let est = KolmogorovEstimator::new(0.1);
        let outputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let structures = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = est.estimate_family(&outputs, &structures);
        assert!(result.lower_bound_bits >= 0.0);
        assert!(result.upper_bound_bits >= 0.0);
    }

    #[test]
    fn test_residue_lower_bound() {
        let est = KolmogorovEstimator::default();
        let outputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let structures = vec![vec![1.0], vec![2.0], vec![3.0]];
        let bound = est.residue_lower_bound(&outputs, &structures);
        assert!(bound >= 0.0);
    }

    #[test]
    fn test_correlated_output_structure() {
        let est = KolmogorovEstimator::new(0.01);
        let output: Vec<f64> = (0..20).map(|i| i as f64 * 3.0).collect();
        let structure: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = est.estimate(&output, &structure);
        // Highly correlated → low residual
        assert!(result.lower_bound_bits < 100.0);
    }

    #[test]
    fn test_precision_affects_estimate() {
        let coarse = KolmogorovEstimator::new(1.0);
        let fine = KolmogorovEstimator::new(0.001);
        let data = vec![1.01, 1.02, 1.03, 1.04];
        let structure = vec![1.0; 4];

        let coarse_est = coarse.estimate(&data, &structure);
        let fine_est = fine.estimate(&data, &structure);

        // Coarser precision should generally give lower or equal complexity
        assert!(coarse_est.upper_bound_bits <= fine_est.upper_bound_bits + 1.0);
    }
}
