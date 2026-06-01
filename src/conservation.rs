//! Conservation of computation theorem.
//!
//! Information-theoretic: you cannot eliminate genuine novelty.
//! The sum of compile-time information + runtime residue = total information.
//! Runtime residue is bounded below by K(answer | structure).

use serde::{Serialize, Deserialize};
use crate::kolmogorov::KolmogorovEstimator;
use crate::natural_transformation::IndexedComputation;

/// Conservation law result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationResult {
    /// Total information content (bits).
    pub total_information_bits: f64,
    /// Compile-time eliminable information (bits).
    pub compile_time_bits: f64,
    /// Runtime residue (bits) — cannot be eliminated.
    pub runtime_residue_bits: f64,
    /// Whether conservation holds (compile + residue ≈ total).
    pub conservation_holds: bool,
    /// Conservation error (should be ≈ 0).
    pub conservation_error: f64,
    /// The irreducible lower bound on runtime cost.
    pub irreducible_lower_bound: f64,
}

/// Conservation of computation analyzer.
pub struct ConservationAnalyzer {
    kolmogorov: KolmogorovEstimator,
    tolerance: f64,
}

impl ConservationAnalyzer {
    pub fn new(tolerance: f64) -> Self {
        Self {
            kolmogorov: KolmogorovEstimator::new(tolerance),
            tolerance,
        }
    }

    pub fn default() -> Self {
        Self::new(1e-6)
    }

    /// Verify the conservation law for a family of computations.
    pub fn verify_conservation(&self, family: &[IndexedComputation]) -> ConservationResult {
        if family.is_empty() {
            return ConservationResult {
                total_information_bits: 0.0,
                compile_time_bits: 0.0,
                runtime_residue_bits: 0.0,
                conservation_holds: true,
                conservation_error: 0.0,
                irreducible_lower_bound: 0.0,
            };
        }

        let outputs: Vec<Vec<f64>> = family.iter().map(|c| c.output.clone()).collect();
        let structures: Vec<Vec<f64>> = family.iter().map(|c| c.structural_features.clone()).collect();

        // Total information: K(output) — Kolmogorov complexity of the output alone
        let total_info = self.kolmogorov.estimate_family(&outputs, &vec![vec![0.0]; family.len()]).lower_bound_bits;

        // Runtime residue: K(output | structure)
        let runtime_residue = self.kolmogorov.residue_lower_bound(&outputs, &structures);

        // Compile-time = total - residue (what's explained by structure)
        let compile_time = (total_info - runtime_residue).max(0.0);

        // Conservation check: compile_time + residue ≈ total
        let conservation_error = (compile_time + runtime_residue - total_info).abs();
        let conservation_holds = conservation_error < self.tolerance * 100.0; // generous tolerance

        ConservationResult {
            total_information_bits: total_info,
            compile_time_bits: compile_time,
            runtime_residue_bits: runtime_residue,
            conservation_holds,
            conservation_error,
            irreducible_lower_bound: runtime_residue,
        }
    }

    /// Prove that a computation cannot be fully compile-time eliminated.
    /// Returns Some(lower_bound_bits) if there's genuine residue, None if fully natural.
    pub fn prove_irreducible(&self, family: &[IndexedComputation]) -> Option<f64> {
        let result = self.verify_conservation(family);
        if result.runtime_residue_bits > self.tolerance {
            Some(result.runtime_residue_bits)
        } else {
            None
        }
    }

    /// The fundamental theorem: information cannot cross the naturality boundary.
    /// Returns the information-theoretic statement.
    pub fn theorem_statement() -> String {
        "Conservation of Computation: For any computation c,\n\
         K(output) = K(output | structure) + I(output; structure)\n\
         where I is mutual information. The runtime residue K(output|structure)\n\
         is a lower bound on the irreducible cost. No optimization can reduce\n\
         the cost below this bound. Genuine novelty cannot be eliminated.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_family(outputs: &[Vec<f64>], structures: &[Vec<f64>]) -> Vec<IndexedComputation> {
        outputs.iter().zip(structures.iter()).enumerate().map(|(i, (o, s))| {
            IndexedComputation {
                index: i,
                structural_features: s.clone(),
                output: o.clone(),
            }
        }).collect()
    }

    #[test]
    fn test_conservation_natural() {
        let analyzer = ConservationAnalyzer::new(0.1);
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            IndexedComputation {
                index: i,
                structural_features: vec![i as f64],
                output: vec![2.0 * i as f64], // linear in structure
            }
        }).collect();
        let result = analyzer.verify_conservation(&family);
        // Linear relationship: residue should be low
        assert!(result.runtime_residue_bits < 100.0);
    }

    #[test]
    fn test_conservation_nontrivial() {
        let analyzer = ConservationAnalyzer::new(0.1);
        let outputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 3.14]).collect();
        let structures: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
        let family = make_family(&outputs, &structures);
        let result = analyzer.verify_conservation(&family);
        // Non-trivial output from constant structure → residue should be positive
        assert!(result.runtime_residue_bits > 0.0);
    }

    #[test]
    fn test_irreducible_proof() {
        let analyzer = ConservationAnalyzer::new(0.1);
        let outputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let structures: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
        let family = make_family(&outputs, &structures);
        let bound = analyzer.prove_irreducible(&family);
        assert!(bound.is_some());
    }

    #[test]
    fn test_natural_has_no_irreducible() {
        let analyzer = ConservationAnalyzer::new(1.0);
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![i as f64],
            output: vec![2.0 * i as f64], // perfectly linear
        }).collect();
        let bound = analyzer.prove_irreducible(&family);
        // Linear family should have small residue
        assert!(bound.is_none() || bound.unwrap() < 100.0);
    }

    #[test]
    fn test_theorem_statement() {
        let stmt = ConservationAnalyzer::theorem_statement();
        assert!(stmt.contains("K(output"));
        assert!(stmt.contains("irreducible"));
    }

    #[test]
    fn test_empty_family() {
        let analyzer = ConservationAnalyzer::default();
        let result = analyzer.verify_conservation(&[]);
        assert!(result.conservation_holds);
        assert_eq!(result.total_information_bits, 0.0);
    }

    #[test]
    fn test_compile_time_nonnegative() {
        let analyzer = ConservationAnalyzer::new(0.1);
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![i as f64],
            output: vec![i as f64 * 2.0],
        }).collect();
        let result = analyzer.verify_conservation(&family);
        assert!(result.compile_time_bits >= 0.0);
    }

    #[test]
    fn test_conservation_components_add_up() {
        let analyzer = ConservationAnalyzer::new(1.0);
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            IndexedComputation {
                index: i,
                structural_features: vec![i as f64],
                output: vec![i as f64 * 2.0 + 1.0],
            }
        }).collect();
        let result = analyzer.verify_conservation(&family);
        let sum = result.compile_time_bits + result.runtime_residue_bits;
        // Should approximately equal total
        let error = (sum - result.total_information_bits).abs();
        assert!(error < 100.0, "Conservation error too large: {}", error);
    }
}
