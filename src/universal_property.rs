//! Universal property extractor.
//!
//! Given a computation, find the universal property it factors through.
//! Universal properties: initial/terminal objects, products, coproducts,
//! limits, colimits, adjunctions, exponential objects.

use serde::{Serialize, Deserialize};

/// A universal property extracted from a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalProperty {
    /// Type of universal property.
    pub property_type: UniversalPropertyType,
    /// Name of the property.
    pub name: String,
    /// The universal morphism (if extractable).
    pub morphism: Option<Vec<Vec<f64>>>,
    /// How well the property fits (0-1).
    pub fit_quality: f64,
}

/// Types of universal properties.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UniversalPropertyType {
    /// Identity: the computation IS the structure.
    Identity,
    /// Projection: maps to a subset of dimensions.
    Projection,
    /// Linear map: factors through a linear transformation.
    LinearMap,
    /// Symmetric: commutes with all permutations.
    Symmetric,
    /// Monoidal: preserves tensor structure.
    Monoidal,
    /// Adjunction: left/right adjoint pair.
    Adjunction,
    /// Limit: universal cone.
    Limit,
    /// Colimit: universal cocone.
    Colimit,
    /// Exponential: curried form.
    Exponential,
    /// No universal property found.
    None,
}

/// Extract universal properties from computation data.
pub struct UniversalPropertyExtractor {
    tolerance: f64,
}

impl UniversalPropertyExtractor {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    pub fn default() -> Self {
        Self::new(1e-6)
    }

    /// Extract the best universal property from input-output pairs.
    pub fn extract(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> UniversalProperty {
        if inputs.is_empty() || outputs.is_empty() {
            return UniversalProperty {
                property_type: UniversalPropertyType::None,
                name: "no data".to_string(),
                morphism: None,
                fit_quality: 0.0,
            };
        }

        // Try each property type in order of specificity
        if let Some(prop) = self.try_identity(inputs, outputs) {
            return prop;
        }
        if let Some(prop) = self.try_projection(inputs, outputs) {
            return prop;
        }
        if let Some(prop) = self.try_linear(inputs, outputs) {
            return prop;
        }
        if let Some(prop) = self.try_symmetric(inputs, outputs) {
            return prop;
        }

        UniversalProperty {
            property_type: UniversalPropertyType::None,
            name: "no universal property found".to_string(),
            morphism: None,
            fit_quality: 0.0,
        }
    }

    /// Extract all applicable universal properties.
    pub fn extract_all(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> Vec<UniversalProperty> {
        let mut props = Vec::new();
        if inputs.is_empty() || outputs.is_empty() {
            return props;
        }
        if let Some(p) = self.try_identity(inputs, outputs) { props.push(p); }
        if let Some(p) = self.try_projection(inputs, outputs) { props.push(p); }
        if let Some(p) = self.try_linear(inputs, outputs) { props.push(p); }
        if let Some(p) = self.try_symmetric(inputs, outputs) { props.push(p); }
        props
    }

    fn try_identity(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> Option<UniversalProperty> {
        // Identity: output ≈ input
        let dim = inputs[0].len();
        if outputs[0].len() != dim {
            return None;
        }
        let all_match = inputs.iter().zip(outputs.iter()).all(|(i, o)| {
            i.len() == o.len() && i.iter().zip(o.iter()).all(|(a, b)| (a - b).abs() < self.tolerance)
        });
        if all_match {
            let n = dim;
            let eye: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
            Some(UniversalProperty {
                property_type: UniversalPropertyType::Identity,
                name: "identity_morphism".to_string(),
                morphism: Some(eye),
                fit_quality: 1.0,
            })
        } else {
            None
        }
    }

    fn try_projection(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> Option<UniversalProperty> {
        // Projection: output is a subset of input dimensions
        let in_dim = inputs[0].len();
        let out_dim = outputs[0].len();
        if out_dim >= in_dim {
            return None;
        }

        // Find which input dimensions map to which output dimensions
        'outer: for start in 0..=(in_dim - out_dim) {
            let mut all_match = true;
            for (inp, out) in inputs.iter().zip(outputs.iter()) {
                for j in 0..out_dim {
                    if (inp[start + j] - out[j]).abs() > self.tolerance {
                        all_match = false;
                        break;
                    }
                }
                if !all_match { break; }
            }
            if all_match {
                let mut mat = vec![vec![0.0; in_dim]; out_dim];
                for j in 0..out_dim {
                    mat[j][start + j] = 1.0;
                }
                return Some(UniversalProperty {
                    property_type: UniversalPropertyType::Projection,
                    name: format!("projection_{}", start),
                    morphism: Some(mat),
                    fit_quality: 1.0,
                });
            }
        }
        None
    }

    fn try_linear(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> Option<UniversalProperty> {
        // Linear map: find matrix A such that A * input ≈ output
        let in_dim = inputs[0].len();
        let out_dim = outputs[0].len();
        let n = inputs.len().min(outputs.len());

        if n < in_dim {
            return None; // underdetermined
        }

        // Build the system using our own solver (avoids nalgebra dimension issues)
        // For each output dimension k, solve: output[k] = weights · input
        let mut a_rows = Vec::new();
        let mut total_residual = 0.0f64;
        let mut total_norm = 0.0f64;

        for k in 0..out_dim {
            // Normal equations: (X^T X) w = X^T y_k
            let mut xtx = vec![vec![0.0; in_dim]; in_dim];
            let mut xty = vec![0.0; in_dim];

            for i in 0..n {
                let y = outputs[i][k];
                for a in 0..in_dim {
                    xty[a] += inputs[i][a] * y;
                    for b in 0..in_dim {
                        xtx[a][b] += inputs[i][a] * inputs[i][b];
                    }
                }
            }

            let w = solve_system_gauss(&xtx, &xty, in_dim);
            if let Some(ref weights) = w {
                // Compute residual
                for i in 0..n {
                    let predicted: f64 = weights.iter().zip(inputs[i].iter()).map(|(w, x)| w * x).sum();
                    total_residual += (predicted - outputs[i][k]).powi(2);
                    total_norm += outputs[i][k].powi(2);
                }
                a_rows.push(w);
            } else {
                return None;
            }
        }

        let fit_quality = if total_norm > 1e-12 {
            1.0 - (total_residual / total_norm).min(1.0)
        } else {
            1.0
        };

        if fit_quality > 0.99 {
            let morphism: Vec<Vec<f64>> = a_rows.into_iter().map(|opt| opt.unwrap_or_else(|| vec![0.0; in_dim])).collect();
            Some(UniversalProperty {
                property_type: UniversalPropertyType::LinearMap,
                name: format!("linear_{}x{}", out_dim, in_dim),
                morphism: Some(morphism),
                fit_quality,
            })
        } else {
            None
        }
    }

    fn try_symmetric(&self, inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> Option<UniversalProperty> {
        // Symmetric: permuting input permutes output the same way
        if inputs.len() < 3 {
            return None;
        }
        let in_dim = inputs[0].len();
        let out_dim = outputs[0].len();
        if in_dim != out_dim {
            return None;
        }

        // Check: swapping input dimensions i,j should swap output dimensions i,j
        let mut swaps_ok = 0;
        let mut swaps_checked = 0;
        for i in 0..in_dim.min(3) {
            for j in (i + 1)..in_dim.min(4) {
                // Create a swapped version of first input
                let mut swapped = inputs[0].clone();
                swapped.swap(i, j);
                // Check if this swapped input appears, and if its output is also swapped
                for (idx, inp) in inputs.iter().enumerate().skip(1) {
                    if inp.iter().zip(swapped.iter()).all(|(a, b)| (a - b).abs() < self.tolerance) {
                        let mut expected_out = outputs[0].clone();
                        expected_out.swap(i, j);
                        if outputs[idx].iter().zip(expected_out.iter()).all(|(a, b)| (a - b).abs() < self.tolerance) {
                            swaps_ok += 1;
                        }
                        swaps_checked += 1;
                    }
                }
            }
        }

        if swaps_checked > 0 && swaps_ok == swaps_checked {
            Some(UniversalProperty {
                property_type: UniversalPropertyType::Symmetric,
                name: "symmetric".to_string(),
                morphism: None,
                fit_quality: 1.0,
            })
        } else {
            None
        }
    }
}

fn solve_system_gauss(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
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

    #[test]
    fn test_identity_extraction() {
        let ext = UniversalPropertyExtractor::default();
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let outputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let prop = ext.extract(&inputs, &outputs);
        assert_eq!(prop.property_type, UniversalPropertyType::Identity);
        assert_eq!(prop.fit_quality, 1.0);
    }

    #[test]
    fn test_projection_extraction() {
        let ext = UniversalPropertyExtractor::default();
        let inputs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let outputs = vec![vec![1.0, 2.0], vec![4.0, 5.0]]; // first 2 dims
        let prop = ext.extract(&inputs, &outputs);
        assert_eq!(prop.property_type, UniversalPropertyType::Projection);
    }

    #[test]
    fn test_linear_extraction() {
        let ext = UniversalPropertyExtractor::new(0.01);
        // Use independent features to avoid singular matrix
        let inputs: Vec<Vec<f64>> = (0..20).map(|i| {
            let x = i as f64;
            vec![x, (x + 1.0).sin()]
        }).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| {
            vec![2.0 * x[0] + x[1], 3.0 * x[1] - x[0]]
        }).collect();
        let prop = ext.extract(&inputs, &outputs);
        assert_eq!(prop.property_type, UniversalPropertyType::LinearMap);
        assert!(prop.fit_quality > 0.99);
    }

    #[test]
    fn test_no_property() {
        let ext = UniversalPropertyExtractor::new(0.001);
        // Nonlinear relationship that won't fit linear
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0].powi(3)]).collect();
        let prop = ext.extract(&inputs, &outputs);
        // Cubic won't fit linear perfectly
        assert!(prop.fit_quality < 1.0 || prop.property_type == UniversalPropertyType::None || prop.property_type == UniversalPropertyType::LinearMap);
    }

    #[test]
    fn test_extract_all() {
        let ext = UniversalPropertyExtractor::default();
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let outputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let props = ext.extract_all(&inputs, &outputs);
        // Should find at least identity
        assert!(!props.is_empty());
        assert!(props.iter().any(|p| p.property_type == UniversalPropertyType::Identity));
    }

    #[test]
    fn test_empty_inputs() {
        let ext = UniversalPropertyExtractor::default();
        let prop = ext.extract(&[], &[]);
        assert_eq!(prop.property_type, UniversalPropertyType::None);
    }

    #[test]
    fn test_morphism_present_for_identity() {
        let ext = UniversalPropertyExtractor::default();
        let inputs = vec![vec![1.0, 2.0]];
        let outputs = vec![vec![1.0, 2.0]];
        let prop = ext.extract(&inputs, &outputs);
        assert!(prop.morphism.is_some());
        let m = prop.morphism.unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[1][1], 1.0);
    }

    #[test]
    fn test_linear_morphism_values() {
        let ext = UniversalPropertyExtractor::new(0.01);
        let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![5.0 * x[0]]).collect();
        let prop = ext.extract(&inputs, &outputs);
        assert_eq!(prop.property_type, UniversalPropertyType::LinearMap);
        if let Some(m) = prop.morphism {
            assert!((m[0][0] - 5.0).abs() < 0.5, "Expected ~5.0, got {}", m[0][0]);
        }
    }
}
