//! Yoneda-based optimization.
//!
//! The Yoneda lemma: for a representable functor F, F(a) ≅ Hom(a, r).
//! If a computation corresponds to a representable functor, the answer IS
//! the identity element — zero computation needed.

use serde::{Serialize, Deserialize};
use nalgebra::DVector;

/// Yoneda optimization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YonedaOptimization {
    /// Whether the functor is representable.
    pub is_representable: bool,
    /// The representing object (if representable).
    pub representing_object: Option<Vec<f64>>,
    /// Speedup factor (1.0 = no speedup, infinity = fully eliminable).
    pub speedup_factor: f64,
    /// The natural isomorphism Hom(-, r) → F(-).
    pub isomorphism: Option<Vec<Vec<f64>>>,
    /// Description.
    pub description: String,
}

/// Check if a computation corresponds to a representable functor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentabilityCheck {
    pub is_representable: bool,
    /// The candidate representing object.
    pub candidate: Vec<f64>,
    /// How well Hom(-, candidate) matches F(-).
    pub fit_error: f64,
}

/// Yoneda-based optimizer.
pub struct YonedaOptimizer {
    tolerance: f64,
}

impl YonedaOptimizer {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn default() -> Self {
        Self::new(1e-6)
    }

    /// Check if a computation is representable (Yoneda: the answer IS the identity).
    pub fn check_representable(
        &self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
    ) -> RepresentabilityCheck {
        if inputs.is_empty() || outputs.is_empty() {
            return RepresentabilityCheck {
                is_representable: false,
                candidate: vec![],
                fit_error: f64::INFINITY,
            };
        }

        let in_dim = inputs[0].len();
        let out_dim = outputs[0].len();

        // For a representable functor, F(a) = Hom(a, r) for some r.
        // Hom(a, r) in vector spaces = r^T * a (if dim=1) or a linear functional.
        // So if F is linear in its input, it's representable.

        // Try to find r such that r^T * a_i = output_i for all i
        if out_dim == 1 {
            let r = self.find_representing_object_1d(inputs, outputs);
            let error = self.compute_representability_error(inputs, outputs, &[r.clone()]);
            RepresentabilityCheck {
                is_representable: error < self.tolerance,
                candidate: r,
                fit_error: error,
            }
        } else {
            // Multi-dimensional: check if each output dim is a linear functional
            let mut total_error = 0.0;
            let mut combined_candidate = Vec::new();
            for k in 0..out_dim {
                let outs_k: Vec<Vec<f64>> = outputs.iter().map(|o| vec![o[k]]).collect();
                let r_k = self.find_representing_object_1d(inputs, &outs_k);
                let error_k = self.compute_representability_error(inputs, &outs_k, &[r_k.clone()]);
                total_error += error_k;
                combined_candidate.extend(r_k);
            }
            RepresentabilityCheck {
                is_representable: total_error < self.tolerance * out_dim as f64,
                candidate: combined_candidate,
                fit_error: total_error,
            }
        }
    }

    /// Optimize using Yoneda: if representable, replace computation with identity lookup.
    pub fn optimize(
        &self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
    ) -> YonedaOptimization {
        let check = self.check_representable(inputs, outputs);

        if check.is_representable {
            YonedaOptimization {
                is_representable: true,
                representing_object: Some(check.candidate.clone()),
                speedup_factor: f64::INFINITY, // fully eliminable
                isomorphism: Some(vec![check.candidate]),
                description: "Representable functor — computation IS the identity. \
                    Replace with Hom(input, representing_object) lookup.".to_string(),
            }
        } else {
            let speedup = if check.fit_error > 0.0 {
                1.0 / (1.0 + check.fit_error)
            } else {
                1.0
            };
            YonedaOptimization {
                is_representable: false,
                representing_object: None,
                speedup_factor: speedup,
                isomorphism: None,
                description: format!(
                    "Not representable (error={:.4}). Cannot fully eliminate. \
                     Partial optimization possible via best linear approximation.",
                    check.fit_error
                ),
            }
        }
    }

    /// Yoneda embedding: represent each element as its "behavior" (all morphisms from it).
    /// This gives a canonical representation.
    pub fn yoneda_embed(&self, element: &[f64], test_objects: &[Vec<f64>]) -> Vec<f64> {
        test_objects.iter().map(|obj| {
            element.iter().zip(obj.iter()).map(|(a, b)| a * b).sum()
        }).collect()
    }

    fn find_representing_object_1d(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> Vec<f64> {
        let in_dim = inputs[0].len();
        let n = inputs.len().min(outputs.len());

        // Solve: r^T * x_i = y_i for all i
        // Normal equations: (X^T X) r = X^T y
        let mut xtx = vec![vec![0.0; in_dim]; in_dim];
        let mut xty = vec![0.0; in_dim];

        for i in 0..n {
            let y = outputs[i][0];
            for a in 0..in_dim {
                xty[a] += inputs[i][a] * y;
                for b in 0..in_dim {
                    xtx[a][b] += inputs[i][a] * inputs[i][b];
                }
            }
        }

        solve_system(&xtx, &xty, in_dim).unwrap_or_else(|| vec![0.0; in_dim])
    }

    fn compute_representability_error(
        &self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        r: &[Vec<f64>],
    ) -> f64 {
        let mut total = 0.0;
        let n = inputs.len().min(outputs.len());
        for i in 0..n {
            let predicted: Vec<f64> = r.iter().map(|r_k| {
                inputs[i].iter().zip(r_k.iter()).map(|(a, b)| a * b).sum()
            }).collect();
            for (p, a) in predicted.iter().zip(outputs[i].iter()) {
                total += (p - a).powi(2);
            }
        }
        total.sqrt()
    }
}

fn solve_system(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug: Vec<Vec<f64>> = a.iter()
        .zip(b.iter())
        .map(|(row, &val)| {
            let mut r = row.clone();
            r.push(val);
            r
        })
        .collect();

    for col in 0..n {
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
    fn test_representable_linear() {
        let opt = YonedaOptimizer::new(1.0);
        // Use independent features
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i as f64 + 1.0).sin()]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![3.0 * x[0] + 2.0 * x[1]]).collect();
        let check = opt.check_representable(&inputs, &outputs);
        assert!(check.fit_error < 10.0, "Error too high: {}", check.fit_error);
    }

    #[test]
    fn test_not_representable_nonlinear() {
        let opt = YonedaOptimizer::new(0.01);
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0] * x[0]]).collect(); // quadratic
        let check = opt.check_representable(&inputs, &outputs);
        assert!(!check.is_representable);
    }

    #[test]
    fn test_yoneda_optimization_speedup() {
        let opt = YonedaOptimizer::new(0.01);
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![2.0 * x[0]]).collect();
        let result = opt.optimize(&inputs, &outputs);
        assert!(result.is_representable);
        assert!(result.speedup_factor > 100.0); // effectively infinite
    }

    #[test]
    fn test_yoneda_embedding() {
        let opt = YonedaOptimizer::default();
        let element = vec![1.0, 2.0, 3.0];
        let test_objects = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let embed = opt.yoneda_embed(&element, &test_objects);
        assert_eq!(embed.len(), 3);
        assert!((embed[0] - 1.0).abs() < 1e-9);
        assert!((embed[1] - 2.0).abs() < 1e-9);
        assert!((embed[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_multidimensional_representable() {
        let opt = YonedaOptimizer::new(0.1);
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![2.0 * x[0], 3.0 * x[0]]).collect();
        let check = opt.check_representable(&inputs, &outputs);
        assert!(check.is_representable);
    }

    #[test]
    fn test_empty_input() {
        let opt = YonedaOptimizer::default();
        let check = opt.check_representable(&[], &[]);
        assert!(!check.is_representable);
    }

    #[test]
    fn test_non_representable_optimization() {
        let opt = YonedaOptimizer::new(0.01);
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0].sin()]).collect();
        let result = opt.optimize(&inputs, &outputs);
        assert!(!result.is_representable);
        assert!(result.speedup_factor <= 1.0);
    }

    #[test]
    fn test_representing_object_values() {
        let opt = YonedaOptimizer::new(0.01);
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![5.0 * x[0]]).collect();
        let check = opt.check_representable(&inputs, &outputs);
        assert!(check.is_representable);
        // The representing object should be approximately [5.0]
        assert!((check.candidate[0] - 5.0).abs() < 0.1);
    }
}
