# lau-naturality-boundary

The Naturality Boundary — where compile-time mathematics ends and runtime computation begins, bounded by Kolmogorov complexity.

## The Theorem

A computation is **compile-time-eliminable** (zero per-instance cost) if and only if it is a **natural transformation** — uniform in the instance, factoring through a universal property. The irreducible runtime residue is bounded by Kolmogorov complexity `K(answer | structure)`.

**You cannot push real information across the boundary.**

## Modules

- **`natural_transformation`** — Detect whether a family of computations is a natural transformation
- **`classifier`** — Classify computation components as compile-time eliminable vs runtime residue
- **`kolmogorov`** — Estimate Kolmogorov complexity K(answer | structure) for runtime residue bounds
- **`universal_property`** — Extract universal properties (identity, projection, linear, symmetric)
- **`yoneda`** — Yoneda-based optimization: representable functors = the answer IS the identity element
- **`parametricity`** — Reynolds' free theorems: what does the type guarantee for free?
- **`residue`** — Residue minimizer: restructure to maximize natural part, minimize residue
- **`conservation`** — Conservation of computation theorem: cannot eliminate genuine novelty
- **`analysis`** — Analyze each lau-* crate for its naturality boundary
- **`case_studies`** — Detailed case studies: Hodge (100% natural), CRDT (100%), warp vote (100%), RL (has residue)

## Case Studies

| Computation | Natural? | Residue | Why |
|---|---|---|---|
| Hodge projection | ✅ 100% | 0 bits | Canonical — factors through image factorization |
| CRDT merge | ✅ 100% | 0 bits | Semilattice operation — associative/commutative/idempotent |
| Warp vote | ✅ 100% | 0 bits | Symmetric function — natural in symmetric group |
| RL policy | ❌ Partial | >0 bits | Environment dynamics are genuine runtime information |

## Usage

```rust
use lau_naturality_boundary::*;

// Detect natural transformations
let detector = natural_transformation::NaturalTransformationDetector::default();
let report = detector.check_naturality(&family);
assert!(report.is_natural);

// Classify compile-time vs runtime
let classifier = classifier::CompileTimeClassifier::default();
let classification = classifier.classify(&family);

// Estimate Kolmogorov complexity of residue
let estimator = kolmogorov::KolmogorovEstimator::default();
let bound = estimator.residue_lower_bound(&outputs, &structures);

// Verify conservation of computation
let analyzer = conservation::ConservationAnalyzer::default();
let result = analyzer.verify_conservation(&family);
assert!(result.conservation_holds);
```

## License

MIT
