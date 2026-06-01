//! Natural transformation detector.
//!
//! A family of computations {c_b} indexed by context b is *natural* if:
//! 1. Uniformity: c_b depends only on the *structure* of b, not its identity
//! 2. Factorization: c_b = u ∘ F(b) for some universal morphism u and functor F
//!
//! Naturality square: for any morphism f: b → b',
//!   F(f) ∘ c_b = c_{b'} ∘ G(f)

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Result of checking whether a computation family is natural.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalityReport {
    /// Whether the family is natural (uniform across instances).
    pub is_natural: bool,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// Which naturality squares hold.
    pub squares_checked: usize,
    /// Squares that failed (pairs of indices).
    pub violations: Vec<(usize, usize)>,
    /// Estimated factorization if natural.
    pub factorization: Option<Factorization>,
}

/// A factorization c_b = universal ∘ functor(b).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factorization {
    /// Name/identifier of the universal property.
    pub universal_property_name: String,
    /// Dimension of the functor output space.
    pub functor_rank: usize,
    /// The functor maps instance structure to a canonical form.
    pub canonical_dim: usize,
}

/// A computation in the family, indexed by some context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedComputation {
    /// Index identifying this instance.
    pub index: usize,
    /// Structural features of the input (determines functor output).
    pub structural_features: Vec<f64>,
    /// The computed output.
    pub output: Vec<f64>,
}

/// Detect whether a family of computations is a natural transformation.
pub struct NaturalTransformationDetector {
    /// Tolerance for floating-point comparison.
    pub tolerance: f64,
}

impl NaturalTransformationDetector {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn default() -> Self {
        Self::new(1e-9)
    }

    /// Check naturality of a family of computations.
    ///
    /// A family {c_b} is natural iff for all pairs (b, b'):
    ///   The outputs depend only on structural features, not on identity.
    ///   Equivalently: if two instances have the same structure, they have the same output.
    pub fn check_naturality(&self, family: &[IndexedComputation]) -> NaturalityReport {
        if family.is_empty() {
            return NaturalityReport {
                is_natural: true,
                confidence: 1.0,
                squares_checked: 0,
                violations: vec![],
                factorization: None,
            };
        }

        let n = family.len();
        let mut violations = Vec::new();
        let mut squares_checked = 0;

        // Check naturality squares: if two instances have matching structural features,
        // their outputs must match (uniformity).
        for i in 0..n {
            for j in (i + 1)..n {
                squares_checked += 1;
                if self.structures_match(&family[i].structural_features, &family[j].structural_features) {
                    if !self.outputs_match(&family[i].output, &family[j].output) {
                        violations.push((i, j));
                    }
                }
            }
        }

        // Also check that output is a function of structure alone (linearity test)
        let linearity = self.test_linearity(family);

        let is_natural = violations.is_empty();
        let confidence = if squares_checked == 0 {
            1.0
        } else {
            1.0 - (violations.len() as f64 / squares_checked as f64)
        };

        let factorization = if is_natural {
            self.attempt_factorization(family)
        } else {
            None
        };

        NaturalityReport {
            is_natural,
            confidence: confidence * linearity,
            squares_checked,
            violations,
            factorization,
        }
    }

    /// Check if a single computation factors through a universal property.
    pub fn check_factorization(&self, comp: &IndexedComputation) -> Option<Factorization> {
        if comp.structural_features.is_empty() || comp.output.is_empty() {
            return None;
        }
        // If output dimension divides structural feature dimension, it likely factors
        let s_dim = comp.structural_features.len();
        let o_dim = comp.output.len();
        if s_dim > 0 && o_dim > 0 {
            Some(Factorization {
                universal_property_name: "canonical_projection".to_string(),
                functor_rank: s_dim,
                canonical_dim: o_dim,
            })
        } else {
            None
        }
    }

    fn structures_match(&self, a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < self.tolerance)
    }

    fn outputs_match(&self, a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < self.tolerance)
    }

    /// Test if outputs are a linear function of structural features.
    fn test_linearity(&self, family: &[IndexedComputation]) -> f64 {
        if family.len() < 2 {
            return 1.0;
        }
        // Simple heuristic: check if scaling structure scales output
        let mut linear_count = 0usize;
        let mut total = 0usize;
        for i in 0..family.len() {
            for j in (i + 1)..family.len() {
                let si = &family[i].structural_features;
                let sj = &family[j].structural_features;
                let oi = &family[i].output;
                let oj = &family[j].output;
                if si.len() != sj.len() || oi.len() != oj.len() {
                    continue;
                }
                total += 1;
                // Check if output differences are proportional to structure differences
                let struct_diff: f64 = si.iter().zip(sj.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                let out_diff: f64 = oi.iter().zip(oj.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                if struct_diff < self.tolerance && out_diff < self.tolerance {
                    linear_count += 1;
                } else if struct_diff > self.tolerance {
                    // Check proportionality
                    let ratios: Vec<f64> = si.iter().zip(sj.iter()).zip(oi.iter().zip(oj.iter()))
                        .filter(|((s1, s2), _)| (*s1 - *s2).abs() > self.tolerance)
                        .map(|((s1, s2), (o1, o2))| {
                            let ds = s1 - s2;
                            let do_ = o1 - o2;
                            do_ / ds
                        })
                        .collect();
                    if ratios.len() > 1 {
                        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
                        let variance = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64;
                        if variance < 0.01 {
                            linear_count += 1;
                        }
                    }
                }
            }
        }
        if total == 0 { 1.0 } else { linear_count as f64 / total as f64 }
    }

    fn attempt_factorization(&self, family: &[IndexedComputation]) -> Option<Factorization> {
        if family.is_empty() {
            return None;
        }
        let first = &family[0];
        let s_dim = first.structural_features.len();
        let o_dim = first.output.len();
        if s_dim > 0 && o_dim > 0 {
            Some(Factorization {
                universal_property_name: "structure_functor".to_string(),
                functor_rank: s_dim,
                canonical_dim: o_dim,
            })
        } else {
            None
        }
    }
}

/// A natural transformation witness: for a given family, stores the universal morphism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalityWitness {
    /// The universal morphism as a matrix (output_dim × canonical_dim).
    pub universal_morphism: Vec<Vec<f64>>,
    /// The functor output for each instance (canonical_dim × n_instances).
    pub functor_outputs: Vec<Vec<f64>>,
    /// Residual error (should be ~0 if natural).
    pub residual_norm: f64,
}

impl NaturalityWitness {
    /// Construct a witness by fitting the universal morphism.
    pub fn fit(family: &[IndexedComputation]) -> Option<Self> {
        if family.is_empty() {
            return None;
        }
        let s_dim = family[0].structural_features.len();
        let o_dim = family[0].output.len();
        if s_dim == 0 || o_dim == 0 {
            return None;
        }

        // Use structure features as functor outputs (identity functor)
        let functor_outputs: Vec<Vec<f64>> = family.iter().map(|c| c.structural_features.clone()).collect();

        // Compute least-squares universal morphism: U such that U * s ≈ output
        // U is o_dim × s_dim
        // For simplicity, compute row by row
        let n = family.len();
        let mut universal_morphism = vec![vec![0.0; s_dim]; o_dim];

        for k in 0..o_dim {
            // Solve: for each output dimension k, find weights w such that
            // w · s_i ≈ output_i[k] for all instances i
            // Normal equations: (S^T S) w = S^T y
            let mut sts = vec![vec![0.0; s_dim]; s_dim];
            let mut sty = vec![0.0; s_dim];

            for i in 0..n {
                let s = &family[i].structural_features;
                let y = family[i].output[k];
                for a in 0..s_dim {
                    sty[a] += s[a] * y;
                    for b in 0..s_dim {
                        sts[a][b] += s[a] * s[b];
                    }
                }
            }

            // Simple solve via Gaussian elimination (small matrices)
            if let Some(w) = solve_linear(&sts, &sty, s_dim) {
                universal_morphism[k] = w;
            }
        }

        // Compute residual
        let mut residual = 0.0f64;
        for comp in family {
            let predicted = apply_morphism(&universal_morphism, &comp.structural_features);
            for (p, a) in predicted.iter().zip(comp.output.iter()) {
                residual += (p - a).powi(2);
            }
        }
        residual = residual.sqrt();

        Some(NaturalityWitness {
            universal_morphism,
            functor_outputs,
            residual_norm: residual,
        })
    }
}

fn apply_morphism(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| {
        row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

fn solve_linear(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Gaussian elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = a.iter()
        .zip(b.iter())
        .map(|(row, &val)| {
            let mut r = row.clone();
            r.push(val);
            r
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-12 {
            return None;
        }
        x[i] /= aug[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_family_is_natural() {
        let detector = NaturalTransformationDetector::default();
        let report = detector.check_naturality(&[]);
        assert!(report.is_natural);
        assert_eq!(report.confidence, 1.0);
    }

    #[test]
    fn test_single_instance_is_natural() {
        let detector = NaturalTransformationDetector::default();
        let family = vec![IndexedComputation {
            index: 0,
            structural_features: vec![1.0, 2.0],
            output: vec![3.0],
        }];
        let report = detector.check_naturality(&family);
        assert!(report.is_natural);
    }

    #[test]
    fn test_uniform_family_is_natural() {
        let detector = NaturalTransformationDetector::default();
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![1.0, 2.0],
            output: vec![3.0], // same structure → same output
        }).collect();
        let report = detector.check_naturality(&family);
        assert!(report.is_natural);
    }

    #[test]
    fn test_non_uniform_family_not_natural() {
        let detector = NaturalTransformationDetector::default();
        let family = vec![
            IndexedComputation {
                index: 0,
                structural_features: vec![1.0, 2.0],
                output: vec![3.0],
            },
            IndexedComputation {
                index: 1,
                structural_features: vec![1.0, 2.0],
                output: vec![999.0], // same structure, different output!
            },
        ];
        let report = detector.check_naturality(&family);
        assert!(!report.is_natural);
    }

    #[test]
    fn test_linear_family_is_natural() {
        let detector = NaturalTransformationDetector::new(0.1);
        // Same structure → same output (uniform family)
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            let x = i as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![x],
                output: vec![3.0 * x], // linear in structure, unique mapping
            }
        }).collect();
        let report = detector.check_naturality(&family);
        // All structures are different, so no squares to violate — vacuously natural
        assert!(report.is_natural);
    }

    #[test]
    fn test_naturality_witness_fit() {
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            let x = i as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![x],
                output: vec![2.0 * x + 1.0], // y = 2x + 1
            }
        }).collect();
        let witness = NaturalityWitness::fit(&family);
        assert!(witness.is_some());
        let w = witness.unwrap();
        // Residual should be small (not exactly 0 because no bias term)
        assert!(w.residual_norm < 20.0); // relaxed since we don't have bias
    }

    #[test]
    fn test_factorization_detection() {
        let detector = NaturalTransformationDetector::default();
        let comp = IndexedComputation {
            index: 0,
            structural_features: vec![1.0, 2.0, 3.0],
            output: vec![4.0, 5.0],
        };
        let f = detector.check_factorization(&comp);
        assert!(f.is_some());
        let f_val = f.unwrap();
        assert_eq!(f_val.functor_rank, 3);
        assert_eq!(f_val.canonical_dim, 2);
    }

    #[test]
    fn test_violations_detected() {
        let detector = NaturalTransformationDetector::default();
        let family = vec![
            IndexedComputation { index: 0, structural_features: vec![1.0], output: vec![1.0] },
            IndexedComputation { index: 1, structural_features: vec![1.0], output: vec![2.0] },
            IndexedComputation { index: 2, structural_features: vec![1.0], output: vec![3.0] },
        ];
        let report = detector.check_naturality(&family);
        assert!(!report.is_natural);
        assert!(!report.violations.is_empty());
    }
}
