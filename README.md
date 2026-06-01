# lau-naturality-boundary

**The Naturality Boundary** — where compile-time mathematics ends and runtime computation begins, bounded by Kolmogorov complexity.

---

## What This Does

This crate implements the theorem that **a computation is compile-time-eliminable if and only if it is a natural transformation** — uniform in the instance and factoring through a universal property. It provides:

- **Natural transformation detection** — check whether a family of computations is natural across instances
- **Compile-time/runtime classifier** — classify computation components as eliminable or residue
- **Kolmogorov complexity estimation** — bound the irreducible runtime residue K(answer | structure)
- **Universal property extraction** — find the categorical structure a computation factors through
- **Yoneda-based optimization** — representable functors → zero-computation answers
- **Parametricity checking** — Reynolds' free theorems from type signatures alone
- **Residue minimization** — restructure computations to maximize the natural part
- **Conservation of computation** — information-theoretic: compile-time + runtime = total
- **Crate analysis** — analyze the lau-* ecosystem for naturality boundaries
- **Case studies** — Hodge projection, CRDT merge, symplectic reduction, consensus commit

The crate contains **81 tests** across 10 modules.

---

## Key Idea

> *You cannot push real information across the boundary.*

The Naturality Boundary theorem states:

1. A computation `c_b` indexed by context `b` is **natural** if it is uniform across instances: `c_b` depends only on the *structure* of `b`, not its identity.

2. Natural computations factor through a **universal property**: `c_b = u ∘ F(b)` for some universal morphism `u` and functor `F`.

3. The irreducible runtime residue equals **K(answer | structure)** — the Kolmogorov complexity of the answer given the structure.

4. By the **conservation law**: compile-time information + runtime residue = total information. You cannot eliminate genuine novelty.

---

## Install

```toml
[dependencies]
lau-naturality-boundary = { git = "https://github.com/SuperInstance/lau-naturality-boundary" }
```

### Dependencies

- `nalgebra` 0.33 — linear algebra for Yoneda/structural analysis
- `serde` 1.x (with `derive`) — serialization

---

## Quick Start

```rust
use lau_naturality_boundary::{
    NaturalTransformationDetector, IndexedComputation,
    CompileTimeClassifier, KolmogorovEstimator,
    ConservationAnalyzer, ResidueMinimizer,
};

// Create a family of computations indexed by context
let family: Vec<IndexedComputation> = vec![
    IndexedComputation {
        index: 0,
        structural_features: vec![1.0, 0.0, 0.0],
        output: vec![2.0, 3.0],
    },
    IndexedComputation {
        index: 1,
        structural_features: vec![0.0, 1.0, 0.0],
        output: vec![5.0, 7.0],
    },
];

// Check naturality
let detector = NaturalTransformationDetector::default();
let report = detector.check_naturality(&family);
println!("Natural: {} (confidence: {})", report.is_natural, report.confidence);

// Classify compile-time vs runtime
let classifier = CompileTimeClassifier::default();
let classification = classifier.classify(&family);
println!("Natural fraction: {}", classification.overall_natural_fraction);

// Estimate Kolmogorov complexity
let kolmo = KolmogorovEstimator::default();
let estimate = kolmo.estimate(&[2.0, 3.0], &[1.0, 0.0, 0.0]);
println!("K(answer|structure): {}-{} bits", estimate.lower_bound_bits, estimate.upper_bound_bits);

// Verify conservation: compile-time + runtime = total
let conservation = ConservationAnalyzer::default();
let result = conservation.verify_conservation(&family);
assert!(result.conservation_holds);

// Minimize residue by restructuring
let minimizer = ResidueMinimizer::default();
let minimized = minimizer.minimize(&family);
println!("Improvement: {} → {}", minimized.natural_fraction_before, minimized.natural_fraction_after);
```

Run all 81 tests:

```bash
cargo test
```

---

## API Reference

### `natural_transformation` — Naturality Detection

| Type/Method | Description |
|------|-------------|
| `NaturalTransformationDetector::new(tol)` | Create with tolerance |
| `.check_naturality(family)` | Check if computation family is natural |
| `NaturalityReport` | is_natural, confidence, squares_checked, violations, factorization |
| `Factorization` | universal_property_name, functor_rank, canonical_dim |
| `IndexedComputation` | index, structural_features, output |

### `classifier` — Compile-Time vs Runtime

| Type/Method | Description |
|------|-------------|
| `CompileTimeClassifier::new(tol)` | Create classifier |
| `.classify(family)` | Full classification |
| `ComponentClass::CompileTime` | Natural — factors through universal property, cost = 0 |
| `ComponentClass::RuntimeResidue` | Not natural — carries genuine information bits |
| `ComponentClass::Partial` | Mixed — natural_fraction describes split |
| `Classification` | components[], overall_natural_fraction, summary |

### `kolmogorov` — Kolmogorov Complexity Estimation

| Type/Method | Description |
|------|-------------|
| `KolmogorovEstimator::new(precision)` | Create estimator |
| `.estimate(output, structure)` | K(output\|structure) lower/upper bounds |
| `.estimate_family(family)` | Batch estimate for a family |
| `KolmogorovEstimate` | lower_bound_bits, upper_bound_bits, method, confidence |

### `universal_property` — Universal Property Extraction

| Type/Method | Description |
|------|-------------|
| `UniversalPropertyExtractor::new(tol)` | Create extractor |
| `.extract(outputs, structures)` | Find the universal property |
| `UniversalPropertyType` | Identity, Projection, LinearMap, Symmetric, Monoidal, Adjunction, Limit, Colimit, Exponential, None |
| `UniversalProperty` | property_type, name, morphism, fit_quality |

### `yoneda` — Yoneda-Based Optimization

| Type/Method | Description |
|------|-------------|
| `YonedaOptimizer::new(tol)` | Create optimizer |
| `.check_representable(observations)` | Check if functor is representable |
| `.optimize(observations)` | Full Yoneda optimization |
| `YonedaOptimization` | is_representable, representing_object, speedup_factor |
| `RepresentabilityCheck` | is_representable, candidate, fit_error |

### `parametricity` — Reynolds' Free Theorems

| Type/Method | Description |
|------|-------------|
| `Type` | Base, Var, Arrow, Product, Sum, List, Generic |
| `ParametricityChecker::new()` | Create checker |
| `.derive_free_theorems(ty)` | Derive free theorems from a type |
| `.check_parametricity(ty, implementations)` | Verify implementations satisfy free theorems |
| `FreeTheorem` | statement, type_signature, strength |
| `TheoremStrength` | Universal, Conditional, Informational |

### `residue` — Residue Minimization

| Type/Method | Description |
|------|-------------|
| `ResidueMinimizer::new(tol)` | Create minimizer |
| `.minimize(family)` | Restructure to maximize natural fraction |
| `MinimizationResult` | restructured, natural_fraction_before/after, improvement, residue_before/after_bits |

### `conservation` — Conservation of Computation

| Type/Method | Description |
|------|-------------|
| `ConservationAnalyzer::new(tol)` | Create analyzer |
| `.verify_conservation(family)` | Verify compile-time + runtime = total |
| `ConservationResult` | total_information_bits, compile_time_bits, runtime_residue_bits, conservation_holds |

### `analysis` — Crate Analysis

| Type/Method | Description |
|------|-------------|
| `CrateAnalyzer::new()` | Create analyzer |
| `.analyze(name, family)` | Full naturality analysis of a crate's computation |
| `CrateAnalysis` | is_natural, natural_fraction, runtime_residue_bits, compile_time_parts, runtime_parts, suggestions |

### `case_studies` — Pre-built Case Studies

| Method | Description |
|------|-------------|
| `CaseStudyRunner::hodge_projection()` | Hodge decomposition: 100% natural |
| `::crdt_merge()` | CRDT merge: natural (idempotent/commutative/associative) |
| `::symplectic_reduction()` | Moment-map reduction: natural (constraint surface) |
| `::consensus_commit()` | Paxos commit: residue from network latency |
| `::run_all()` | Run all case studies |
| `CaseStudy` | name, is_fully_natural, natural_fraction, residue_bits, universal_property, summary |

---

## How It Works

### The Naturality Check

Given a family of computations `{c_b}` indexed by context `b`, the detector checks:

1. **Naturality squares**: For morphisms f: b → b', verify F(f) ∘ c_b = c_{b'} ∘ G(f)
2. **Uniformity**: Output depends only on structural features, not instance identity
3. **Factorization**: Can we find c_b = u ∘ F(b)?

If all squares commute and factorization succeeds, the family is natural → compile-time eliminable.

### Kolmogorov Estimation

Since true Kolmogorov complexity is uncomputable, the estimator uses:

- **Compression-based methods**: RLE, entropy, mutual information between output and structure
- **Structural analysis**: Linear/polynomial structure detection
- **Lower bounds**: Empirical entropy of the output conditioned on the structure

### Yoneda Optimization

The Yoneda lemma states: for a representable functor F, F(a) ≅ Hom(a, r) for some representing object r.

If a computation corresponds to a representable functor:
- The answer IS the identity element
- Speedup = ∞ (fully eliminable)
- The representing object is the "canonical form"

### Parametricity (Free Theorems)

From a type signature alone, Reynolds' parametricity derives "free theorems" — properties that must hold for all implementations:

- `forall a. a → a` → must be identity
- `forall a. [a] → [a]` → must commute with map
- `forall a. (a → a) → [a] → [a]` → must apply function uniformly

These are compile-time guarantees that constrain runtime behavior.

---

## The Math

### Natural Transformations

Given functors F, G: C → D, a natural transformation η: F ⇒ G assigns to each object c a morphism η_c: F(c) → G(c) such that for every f: c → c':

```
η_{c'} ∘ F(f) = G(f) ∘ η_c
```

This is the **naturality square**. If a computation family satisfies this, it is "uniform in the instance" — the structure determines the answer.

### Kolmogorov Complexity

K(x | y) = length of shortest program that outputs x given y as input.

Key properties:
- **Uncomputable**: No algorithm computes K exactly (Chaitin's theorem)
- **Upper bounded**: Any compression algorithm gives an upper bound
- **Invariant**: Up to an additive constant, K is independent of the programming language

### Conservation of Computation

```
I(total) = I(compile-time) + I(runtime)
I(runtime) ≥ K(answer | structure)
```

You cannot eliminate genuine information. The compile-time part is determined by structure; the runtime part carries genuine novelty. Their sum is conserved.

### Yoneda Lemma

For a locally small category C and functor F: C^op → Set:

```
F(a) ≅ Nat(Hom(-, a), F)
```

If F is representable (F ≅ Hom(-, r)), then evaluating F(a) reduces to looking up an element of Hom(a, r) — the answer is the identity morphism on r.

### Reynolds' Parametricity

For a polymorphic function `f: ∀α. T(α)`, the parametricity theorem states:

```
forall R ⊆ A × B. (a, b) ∈ R ⟹ (f_A(a), f_B(b)) ∈ T(R)
```

where R is any relation between types A and B, and T(R) is the "relation lifting" through type constructor T. This produces "free theorems" — properties that cost nothing to verify because they follow from the type alone.

---

## License

MIT
