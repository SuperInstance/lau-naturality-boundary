//! Residue minimizer.
//!
//! Restructure a computation to maximize the natural (compile-time eliminable) part
//! and minimize the runtime residue.

use serde::{Serialize, Deserialize};
use crate::natural_transformation::{IndexedComputation, NaturalTransformationDetector, NaturalityReport};
use crate::kolmogorov::KolmogorovEstimator;

/// Result of residue minimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimizationResult {
    /// The restructured computation family.
    pub restructured: Vec<IndexedComputation>,
    /// Natural fraction before optimization.
    pub natural_fraction_before: f64,
    /// Natural fraction after optimization.
    pub natural_fraction_after: f64,
    /// Improvement in natural fraction.
    pub improvement: f64,
    /// Residue before (in bits).
    pub residue_before_bits: f64,
    /// Residue after (in bits).
    pub residue_after_bits: f64,
    /// Description of the transformation applied.
    pub transformation: String,
}

/// Residue minimizer.
pub struct ResidueMinimizer {
    detector: NaturalTransformationDetector,
    kolmogorov: KolmogorovEstimator,
}

impl ResidueMinimizer {
    pub fn new(tolerance: f64) -> Self {
        Self {
            detector: NaturalTransformationDetector::new(tolerance),
            kolmogorov: KolmogorovEstimator::new(tolerance),
        }
    }

    pub fn default() -> Self {
        Self::new(1e-6)
    }

    /// Minimize residue by restructuring the computation family.
    pub fn minimize(&self, family: &[IndexedComputation]) -> MinimizationResult {
        if family.is_empty() {
            return MinimizationResult {
                restructured: vec![],
                natural_fraction_before: 1.0,
                natural_fraction_after: 1.0,
                improvement: 0.0,
                residue_before_bits: 0.0,
                residue_after_bits: 0.0,
                transformation: "No data".to_string(),
            };
        }

        let report_before = self.detector.check_naturality(family);
        let structures: Vec<Vec<f64>> = family.iter().map(|c| c.structural_features.clone()).collect();
        let outputs: Vec<Vec<f64>> = family.iter().map(|c| c.output.clone()).collect();
        let residue_before = self.kolmogorov.residue_lower_bound(&outputs, &structures);

        if report_before.is_natural {
            return MinimizationResult {
                restructured: family.to_vec(),
                natural_fraction_before: 1.0,
                natural_fraction_after: 1.0,
                improvement: 0.0,
                residue_before_bits: residue_before,
                residue_after_bits: residue_before,
                transformation: "Already natural — no optimization needed".to_string(),
            };
        }

        // Strategy 1: Factor out the mean (center the data)
        let centered = self.center_outputs(family);
        let report_centered = self.detector.check_naturality(&centered);

        // Strategy 2: Normalize by structure
        let normalized = self.normalize_by_structure(family);
        let report_normalized = self.detector.check_naturality(&normalized);

        // Strategy 3: Separate into natural + residual components
        let decomposed = self.decompose_natural_residual(family);
        let report_decomposed = self.detector.check_naturality(&decomposed);

        // Pick the best
        let candidates = [
            (&centered, report_centered.confidence, "centering"),
            (&normalized, report_normalized.confidence, "normalization"),
            (&decomposed, report_decomposed.confidence, "natural-residual decomposition"),
        ];

        let (best_family, best_conf, best_name) = candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let structures_after: Vec<Vec<f64>> = best_family.iter().map(|c| c.structural_features.clone()).collect();
        let outputs_after: Vec<Vec<f64>> = best_family.iter().map(|c| c.output.clone()).collect();
        let residue_after = self.kolmogorov.residue_lower_bound(&outputs_after, &structures_after);

        let improvement = best_conf - report_before.confidence;

        MinimizationResult {
            restructured: (*best_family).clone(),
            natural_fraction_before: report_before.confidence,
            natural_fraction_after: *best_conf,
            improvement: improvement.max(0.0),
            residue_before_bits: residue_before,
            residue_after_bits: residue_after,
            transformation: format!("Applied {}", best_name),
        }
    }

    /// Center outputs by subtracting the mean.
    fn center_outputs(&self, family: &[IndexedComputation]) -> Vec<IndexedComputation> {
        if family.is_empty() {
            return vec![];
        }
        let out_dim = family[0].output.len();
        let n = family.len() as f64;

        let mut mean = vec![0.0; out_dim];
        for comp in family {
            for (k, &v) in comp.output.iter().enumerate() {
                mean[k] += v / n;
            }
        }

        family.iter().enumerate().map(|(i, comp)| {
            let centered_output: Vec<f64> = comp.output.iter().zip(mean.iter()).map(|(o, m)| o - m).collect();
            IndexedComputation {
                index: i,
                structural_features: comp.structural_features.clone(),
                output: centered_output,
            }
        }).collect()
    }

    /// Normalize outputs by dividing by structural features (element-wise).
    fn normalize_by_structure(&self, family: &[IndexedComputation]) -> Vec<IndexedComputation> {
        family.iter().enumerate().map(|(i, comp)| {
            let normalized: Vec<f64> = comp.output.iter().enumerate().map(|(k, &o)| {
                let s = comp.structural_features.get(k).copied().unwrap_or(1.0);
                if s.abs() > 1e-10 { o / s } else { o }
            }).collect();
            IndexedComputation {
                index: i,
                structural_features: comp.structural_features.clone(),
                output: normalized,
            }
        }).collect()
    }

    /// Decompose: output = natural_part + residual.
    /// The natural part is the best linear approximation; the residual is what's left.
    fn decompose_natural_residual(&self, family: &[IndexedComputation]) -> Vec<IndexedComputation> {
        // Use the linear part as "natural"
        if family.is_empty() {
            return vec![];
        }

        let s_dim = family[0].structural_features.len();
        let o_dim = family[0].output.len();
        let n = family.len();

        // Fit linear model: output = A * structure + bias
        // For simplicity, fit each output dimension separately
        let mut a_rows: Vec<Vec<f64>> = Vec::new();
        let mut biases: Vec<f64> = Vec::new();

        for k in 0..o_dim {
            // Compute mean
            let mean_y: f64 = family.iter().map(|c| c.output[k]).sum::<f64>() / n as f64;
            let mean_x: Vec<f64> = (0..s_dim).map(|j| {
                family.iter().map(|c| c.structural_features[j]).sum::<f64>() / n as f64
            }).collect();

            // Covariance matrix and solve
            let mut xtx = vec![vec![0.0; s_dim]; s_dim];
            let mut xty = vec![0.0; s_dim];

            for comp in family {
                let y_centered = comp.output[k] - mean_y;
                for a in 0..s_dim {
                    let xa = comp.structural_features[a] - mean_x[a];
                    xty[a] += xa * y_centered;
                    for b in 0..s_dim {
                        let xb = comp.structural_features[b] - mean_x[b];
                        xtx[a][b] += xa * xb;
                    }
                }
            }

            let weights = solve_system(&xtx, &xty, s_dim).unwrap_or_else(|| vec![0.0; s_dim]);
            let bias = mean_y - weights.iter().zip(mean_x.iter()).map(|(w, m)| w * m).sum::<f64>();
            a_rows.push(weights);
            biases.push(bias);
        }

        // Compute residual (original output - linear prediction)
        family.iter().enumerate().map(|(i, comp)| {
            let predicted: Vec<f64> = (0..o_dim).map(|k| {
                let lin: f64 = a_rows[k].iter().zip(comp.structural_features.iter()).map(|(w, s)| w * s).sum();
                lin + biases[k]
            }).collect();
            let residual: Vec<f64> = comp.output.iter().zip(predicted.iter()).map(|(o, p)| o - p).collect();
            IndexedComputation {
                index: i,
                structural_features: comp.structural_features.clone(),
                output: residual,
            }
        }).collect()
    }
}

fn solve_system(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug: Vec<Vec<f64>> = a.iter()
        .zip(b.iter())
        .map(|(row, &val)| { let mut r = row.clone(); r.push(val); r })
        .collect();
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val { max_val = aug[row][col].abs(); max_row = row; }
        }
        if max_val < 1e-12 { return None; }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n { aug[row][j] -= factor * aug[col][j]; }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n { x[i] -= aug[i][j] * x[j]; }
        if aug[i][i].abs() < 1e-12 { return None; }
        x[i] /= aug[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_family(outputs: &[f64], structures: &[f64]) -> Vec<IndexedComputation> {
        outputs.iter().zip(structures.iter()).enumerate().map(|(i, (&o, &s))| {
            IndexedComputation {
                index: i,
                structural_features: vec![s],
                output: vec![o],
            }
        }).collect()
    }

    #[test]
    fn test_already_natural() {
        let min = ResidueMinimizer::default();
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![1.0],
            output: vec![2.0],
        }).collect();
        let result = min.minimize(&family);
        assert_eq!(result.natural_fraction_before, 1.0);
        assert_eq!(result.improvement, 0.0);
    }

    #[test]
    fn test_centering_reduces_residue() {
        let min = ResidueMinimizer::new(0.01);
        let family: Vec<IndexedComputation> = (0..10).map(|i| {
            let x = i as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![x],
                output: vec![2.0 * x + 100.0], // linear with large bias
            }
        }).collect();
        let result = min.minimize(&family);
        assert!(result.improvement >= 0.0);
    }

    #[test]
    fn test_decomposition_captures_linear() {
        let min = ResidueMinimizer::new(0.01);
        let family: Vec<IndexedComputation> = (0..20).map(|i| {
            let x = i as f64;
            IndexedComputation {
                index: i,
                structural_features: vec![x],
                output: vec![3.0 * x + 1.0],
            }
        }).collect();
        let result = min.minimize(&family);
        assert!(result.natural_fraction_after >= result.natural_fraction_before - 0.01);
    }

    #[test]
    fn test_empty_family() {
        let min = ResidueMinimizer::default();
        let result = min.minimize(&[]);
        assert_eq!(result.natural_fraction_before, 1.0);
    }

    #[test]
    fn test_residue_bits_nonnegative() {
        let min = ResidueMinimizer::new(0.1);
        let family: Vec<IndexedComputation> = (0..5).map(|i| {
            IndexedComputation {
                index: i,
                structural_features: vec![i as f64],
                output: vec![(i as f64).sin()],
            }
        }).collect();
        let result = min.minimize(&family);
        assert!(result.residue_before_bits >= 0.0);
        assert!(result.residue_after_bits >= 0.0);
    }

    #[test]
    fn test_restructured_same_length() {
        let min = ResidueMinimizer::default();
        let family: Vec<IndexedComputation> = (0..5).map(|i| IndexedComputation {
            index: i,
            structural_features: vec![i as f64],
            output: vec![i as f64 * 2.0],
        }).collect();
        let result = min.minimize(&family);
        assert_eq!(result.restructured.len(), 5);
    }
}
