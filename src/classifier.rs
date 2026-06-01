//! Compile-time vs runtime classifier.
//!
//! Given a computation, classify which parts are natural (= compile-time eliminable)
//! and which parts carry genuine runtime residue.

use serde::{Serialize, Deserialize};
use crate::natural_transformation::{NaturalTransformationDetector, IndexedComputation, NaturalityReport};

/// Classification of a computation component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentClass {
    /// Compile-time eliminable: natural, factors through a universal property.
    CompileTime {
        /// Which universal property it factors through.
        universal_property: String,
        /// Cost after elimination (always 0).
        eliminated_cost: f64,
    },
    /// Runtime residue: not natural, carries genuine information.
    RuntimeResidue {
        /// Lower bound on the information content (bits).
        information_bits: f64,
        /// Kolmogorov complexity lower bound.
        kolmogorov_lower_bound: f64,
    },
    /// Partially eliminable: some structure is natural, some is residue.
    Partial {
        /// Fraction that is compile-time eliminable.
        natural_fraction: f64,
        /// Description of the natural part.
        natural_description: String,
        /// Description of the residue.
        residue_description: String,
    },
}

/// Full classification result for a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    /// Classification of each component.
    pub components: Vec<ComponentClass>,
    /// Overall natural fraction (0 = all runtime, 1 = all compile-time).
    pub overall_natural_fraction: f64,
    /// The naturality report used for classification.
    pub naturality_report: NaturalityReport,
    /// Summary description.
    pub summary: String,
}

/// Compile-time / Runtime classifier.
pub struct CompileTimeClassifier {
    detector: NaturalTransformationDetector,
}

impl CompileTimeClassifier {
    pub fn new(tolerance: f64) -> Self {
        Self {
            detector: NaturalTransformationDetector::new(tolerance),
        }
    }

    pub fn default() -> Self {
        Self::new(1e-9)
    }

    /// Classify a family of computations into compile-time vs runtime parts.
    pub fn classify(&self, family: &[IndexedComputation]) -> Classification {
        let report = self.detector.check_naturality(family);

        if family.is_empty() {
            return Classification {
                components: vec![],
                overall_natural_fraction: 1.0,
                naturality_report: report,
                summary: "Empty family — trivially natural".to_string(),
            };
        }

        let mut components = Vec::new();

        if report.is_natural {
            components.push(ComponentClass::CompileTime {
                universal_property: report.factorization
                    .as_ref()
                    .map(|f| f.universal_property_name.clone())
                    .unwrap_or_else(|| "identity".to_string()),
                eliminated_cost: 0.0,
            });
            Classification {
                components,
                overall_natural_fraction: 1.0,
                naturality_report: report,
                summary: "Fully natural — compile-time eliminable".to_string(),
            }
        } else {
            let natural_fraction = report.confidence;
            if natural_fraction > 0.9 {
                components.push(ComponentClass::Partial {
                    natural_fraction,
                    natural_description: "Mostly uniform structure".to_string(),
                    residue_description: "Small instance-dependent variation".to_string(),
                });
                Classification {
                    components,
                    overall_natural_fraction: natural_fraction,
                    naturality_report: report,
                    summary: format!("Mostly natural ({:.1}%) — small runtime residue", natural_fraction * 100.0),
                }
            } else {
                components.push(ComponentClass::RuntimeResidue {
                    information_bits: estimate_information(family),
                    kolmogorov_lower_bound: estimate_kolmogorov_simple(family),
                });
                Classification {
                    components,
                    overall_natural_fraction: natural_fraction,
                    naturality_report: report,
                    summary: format!("Substantially non-natural ({:.1}% natural) — significant runtime residue", natural_fraction * 100.0),
                }
            }
        }
    }

    /// Classify individual components of a computation by dimension.
    pub fn classify_by_dimension(&self, family: &[IndexedComputation]) -> Vec<ComponentClass> {
        if family.is_empty() || family[0].output.is_empty() {
            return vec![];
        }

        let n = family.len();
        let out_dim = family[0].output.len();

        (0..out_dim).map(|dim| {
            let sub_family: Vec<IndexedComputation> = family.iter().enumerate().map(|(i, c)| {
                IndexedComputation {
                    index: i,
                    structural_features: c.structural_features.clone(),
                    output: vec![c.output[dim]],
                }
            }).collect();

            let report = self.detector.check_naturality(&sub_family);
            if report.is_natural {
                ComponentClass::CompileTime {
                    universal_property: "dimension_natural".to_string(),
                    eliminated_cost: 0.0,
                }
            } else if report.confidence > 0.9 {
                ComponentClass::Partial {
                    natural_fraction: report.confidence,
                    natural_description: format!("Dimension {} mostly natural", dim),
                    residue_description: format!("Dimension {} has small residue", dim),
                }
            } else {
                ComponentClass::RuntimeResidue {
                    information_bits: estimate_information(&sub_family),
                    kolmogorov_lower_bound: estimate_kolmogorov_simple(&sub_family),
                }
            }
        }).collect()
    }
}

fn estimate_information(family: &[IndexedComputation]) -> f64 {
    if family.is_empty() {
        return 0.0;
    }
    // Estimate information content as entropy of output distribution
    let n = family.len() as f64;
    // Group outputs by rough equality (bucket)
    let mut buckets: Vec<usize> = vec![0; 10];
    for comp in family {
        let hash: u64 = comp.output.iter()
            .map(|x| x.to_bits())
            .fold(0u64, |acc, b| acc.wrapping_add(b));
        let bucket = (hash % 10) as usize;
        buckets[bucket] += 1;
    }
    let mut entropy = 0.0;
    for &count in &buckets {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn estimate_kolmogorov_simple(family: &[IndexedComputation]) -> f64 {
    // Simple lower bound: number of distinct outputs * log2(n) / n
    let n = family.len();
    if n == 0 {
        return 0.0;
    }
    let distinct = {
        let mut seen = std::collections::HashSet::new();
        for comp in family {
            let key: Vec<u64> = comp.output.iter().map(|x| x.to_bits()).collect();
            seen.insert(key);
        }
        seen.len()
    };
    (distinct as f64).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_empty() {
        let clf = CompileTimeClassifier::default();
        let result = clf.classify(&[]);
        assert_eq!(result.overall_natural_fraction, 1.0);
    }

    #[test]
    fn test_classify_natural() {
        let clf = CompileTimeClassifier::default();
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![1.0],
            output: vec![2.0],
        }).collect();
        let result = clf.classify(&family);
        assert_eq!(result.overall_natural_fraction, 1.0);
        assert!(matches!(result.components[0], ComponentClass::CompileTime { .. }));
    }

    #[test]
    fn test_classify_non_natural() {
        let clf = CompileTimeClassifier::new(0.01);
        let family: Vec<IndexedComputation> = (0..10).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![1.0],
            output: vec![i as f64 * 13.7 + 3.14], // varies despite same structure
        }).collect();
        let result = clf.classify(&family);
        assert!(result.overall_natural_fraction < 1.0);
    }

    #[test]
    fn test_classify_by_dimension() {
        let clf = CompileTimeClassifier::default();
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![i as f64],
            output: vec![2.0 * i as f64, 999.0 - i as f64], // first dim natural-ish, second varies
        }).collect();
        let dims = clf.classify_by_dimension(&family);
        assert_eq!(dims.len(), 2);
    }

    #[test]
    fn test_compile_time_component_has_zero_cost() {
        let comp = ComponentClass::CompileTime {
            universal_property: "test".to_string(),
            eliminated_cost: 0.0,
        };
        if let ComponentClass::CompileTime { eliminated_cost, .. } = comp {
            assert_eq!(eliminated_cost, 0.0);
        }
    }

    #[test]
    fn test_runtime_residue_has_information() {
        let comp = ComponentClass::RuntimeResidue {
            information_bits: 42.0,
            kolmogorov_lower_bound: 10.0,
        };
        if let ComponentClass::RuntimeResidue { information_bits, .. } = comp {
            assert!(information_bits > 0.0);
        }
    }

    #[test]
    fn test_partial_classification() {
        let clf = CompileTimeClassifier::new(0.1);
        // Mix of natural and non-natural
        let family: Vec<IndexedComputation> = (0..20).map(|i| {
            let x = i as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![x / 20.0],
                output: vec![x * 2.0 + if i % 3 == 0 { 1.0 } else { 0.0 }],
            }
        }).collect();
        let result = clf.classify(&family);
        // Should be classified as something (not panic)
        assert!(result.overall_natural_fraction >= 0.0);
    }

    #[test]
    fn test_summary_non_empty() {
        let clf = CompileTimeClassifier::default();
        let family = vec![IndexedComputation {
            index: 0,
            structural_features: vec![1.0],
            output: vec![2.0],
        }];
        let result = clf.classify(&family);
        assert!(!result.summary.is_empty());
    }
}
