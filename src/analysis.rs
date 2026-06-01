//! Analysis of existing lau-* crates for their naturality boundary.

use serde::{Serialize, Deserialize};
use crate::natural_transformation::{NaturalTransformationDetector, IndexedComputation, NaturalityReport};
use crate::classifier::CompileTimeClassifier;
use crate::kolmogorov::KolmogorovEstimator;

/// Naturality boundary analysis of a crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateAnalysis {
    /// Crate name.
    pub crate_name: String,
    /// Description of the core computation.
    pub computation_description: String,
    /// Is the core computation natural?
    pub is_natural: bool,
    /// Natural fraction (0-1).
    pub natural_fraction: f64,
    /// Runtime residue in bits.
    pub runtime_residue_bits: f64,
    /// Which parts are compile-time eliminable.
    pub compile_time_parts: Vec<String>,
    /// Which parts have runtime residue.
    pub runtime_parts: Vec<String>,
    /// Universal property it factors through (if any).
    pub universal_property: Option<String>,
    /// Optimization suggestions.
    pub suggestions: Vec<String>,
}

/// Analyze the lau-* crate ecosystem.
pub struct CrateAnalyzer {
    detector: NaturalTransformationDetector,
    classifier: CompileTimeClassifier,
    kolmogorov: KolmogorovEstimator,
}

impl CrateAnalyzer {
    pub fn new() -> Self {
        Self {
            detector: NaturalTransformationDetector::default(),
            classifier: CompileTimeClassifier::default(),
            kolmogorov: KolmogorovEstimator::default(),
        }
    }

    /// Analyze a specific computation pattern from a crate.
    pub fn analyze(&self, crate_name: &str, family: &[IndexedComputation]) -> CrateAnalysis {
        let report = self.detector.check_naturality(family);
        let classification = self.classifier.classify(family);
        let structures: Vec<Vec<f64>> = family.iter().map(|c| c.structural_features.clone()).collect();
        let outputs: Vec<Vec<f64>> = family.iter().map(|c| c.output.clone()).collect();
        let residue = self.kolmogorov.residue_lower_bound(&outputs, &structures);

        let mut suggestions = Vec::new();
        if report.is_natural {
            suggestions.push("Fully natural — can be completely compile-time eliminated".to_string());
        } else {
            suggestions.push(format!("Runtime residue: {:.2} bits — cannot be eliminated", residue));
            if report.confidence > 0.8 {
                suggestions.push("Mostly natural — consider factoring out the natural part".to_string());
            }
            suggestions.push("Consider restructuring to maximize the natural component".to_string());
        }

        let universal_property = report.factorization.as_ref().map(|f| f.universal_property_name.clone());

        CrateAnalysis {
            crate_name: crate_name.to_string(),
            computation_description: format!("{} computation family ({} instances)", crate_name, family.len()),
            is_natural: report.is_natural,
            natural_fraction: classification.overall_natural_fraction,
            runtime_residue_bits: residue,
            compile_time_parts: if report.is_natural { vec!["entire computation".to_string()] } else { vec!["structural component".to_string()] },
            runtime_parts: if report.is_natural { vec![] } else { vec!["instance-dependent variation".to_string()] },
            universal_property,
            suggestions,
        }
    }

    /// Analyze the known lau-* crates (static analysis based on known properties).
    pub fn analyze_known_crates(&self) -> Vec<CrateAnalysis> {
        vec![
            self.analyze_hodge_projection(),
            self.analyze_crdt_merge(),
            self.analyze_warp_vote(),
            self.analyze_rl_policy(),
            self.analyze_tensor_ops(),
            self.analyze_graph_traversal(),
        ]
    }

    fn analyze_hodge_projection(&self) -> CrateAnalysis {
        // Hodge projection: project onto harmonic forms. 100% natural.
        // The Hodge decomposition is canonical — it factors through the universal
        // property of the image factorization (ker ⊕ im → V).
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            let x = i as f64 / 10.0;
            IndexedComputation {
                index: i,
                structural_features: vec![x, x * x],
                output: vec![x, x * x], // projection = identity on harmonic forms
            }
        }).collect();
        self.analyze("lau-hodge", &family)
    }

    fn analyze_crdt_merge(&self) -> CrateAnalysis {
        // CRDT merge: associative, commutative, idempotent. 100% natural.
        // The merge operation is a semilattice operation — natural in both arguments.
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            let a = i as f64;
            let b = (10 - i) as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![a, b],
                output: vec![a.max(b)], // max is the semilattice merge
            }
        }).collect();
        self.analyze("lau-crdt", &family)
    }

    fn analyze_warp_vote(&self) -> CrateAnalysis {
        // Warp vote: all-reduce operations. 100% natural.
        // The vote is a symmetric function — natural in the permutation group.
        let family: Vec<IndexedComputation> = (0..5).map(|i| {
            let base = vec![1.0, 2.0, 3.0, 4.0];
            let rotated: Vec<f64> = base.iter().cycle().skip(i).take(4).copied().collect();
            IndexedComputation {
                index: i,
                structural_features: rotated.clone(),
                output: vec![10.0], // sum is symmetric
            }
        }).collect();
        self.analyze("lau-warp", &family)
    }

    fn analyze_rl_policy(&self) -> CrateAnalysis {
        // RL policy: depends on actual reward signal. Has genuine residue.
        // The optimal policy depends on the environment — genuine runtime information.
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            IndexedComputation {
                index: i,
                structural_features: vec![1.0, 0.0], // same structure
                output: vec![(i as f64 * 0.73).sin(), (i as f64 * 1.31).cos()], // different outputs
            }
        }).collect();
        let mut analysis = self.analyze("lau-rl", &family);
        analysis.compile_time_parts = vec!["action space structure".to_string(), "policy architecture".to_string()];
        analysis.runtime_parts = vec!["environment dynamics (reward)".to_string(), "optimal action selection".to_string()];
        analysis.suggestions.push("RL policy has genuine residue: environment dynamics are runtime information".to_string());
        analysis.suggestions.push("Only the policy architecture is compile-time eliminable".to_string());
        analysis
    }

    fn analyze_tensor_ops(&self) -> CrateAnalysis {
        // Tensor contractions: linear, natural.
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            let x = i as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![x, 2.0 * x],
                output: vec![3.0 * x], // linear contraction
            }
        }).collect();
        self.analyze("lau-tensor", &family)
    }

    fn analyze_graph_traversal(&self) -> CrateAnalysis {
        // Graph algorithms: structure-dependent, partially natural.
        let family: Vec<IndexedComputation> = (0..5).map(|i| {
            IndexedComputation {
                index: i,
                structural_features: vec![i as f64, (i + 1) as f64],
                output: vec![(i * 2 + 1) as f64], // depends on graph structure
            }
        }).collect();
        let mut analysis = self.analyze("lau-graph", &family);
        analysis.compile_time_parts = vec!["traversal order template".to_string()];
        analysis.runtime_parts = vec!["actual graph adjacency".to_string()];
        analysis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hodge_100_percent_natural() {
        let analyzer = CrateAnalyzer::new();
        let analyses = analyzer.analyze_known_crates();
        let hodge = analyses.iter().find(|a| a.crate_name == "lau-hodge").unwrap();
        assert!(hodge.is_natural);
        assert_eq!(hodge.natural_fraction, 1.0);
    }

    #[test]
    fn test_crdt_100_percent_natural() {
        let analyzer = CrateAnalyzer::new();
        let analyses = analyzer.analyze_known_crates();
        let crdt = analyses.iter().find(|a| a.crate_name == "lau-crdt").unwrap();
        assert!(crdt.is_natural);
    }

    #[test]
    fn test_warp_100_percent_natural() {
        let analyzer = CrateAnalyzer::new();
        let analyses = analyzer.analyze_known_crates();
        let warp = analyses.iter().find(|a| a.crate_name == "lau-warp").unwrap();
        assert!(warp.is_natural);
    }

    #[test]
    fn test_rl_has_residue() {
        let analyzer = CrateAnalyzer::new();
        let analyses = analyzer.analyze_known_crates();
        let rl = analyses.iter().find(|a| a.crate_name == "lau-rl").unwrap();
        assert!(!rl.is_natural);
        assert!(!rl.runtime_parts.is_empty());
    }

    #[test]
    fn test_tensor_natural() {
        let analyzer = CrateAnalyzer::new();
        let analyses = analyzer.analyze_known_crates();
        let tensor = analyses.iter().find(|a| a.crate_name == "lau-tensor").unwrap();
        assert!(tensor.is_natural);
    }

    #[test]
    fn test_all_crates_analyzed() {
        let analyzer = CrateAnalyzer::new();
        let analyses = analyzer.analyze_known_crates();
        assert!(analyses.len() >= 5);
        for a in &analyses {
            assert!(!a.crate_name.is_empty());
            assert!(!a.suggestions.is_empty());
        }
    }

    #[test]
    fn test_custom_analysis() {
        let analyzer = CrateAnalyzer::new();
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![i as f64],
            output: vec![i as f64 * 2.0],
        }).collect();
        let analysis = analyzer.analyze("custom-crate", &family);
        assert_eq!(analysis.crate_name, "custom-crate");
    }
}
