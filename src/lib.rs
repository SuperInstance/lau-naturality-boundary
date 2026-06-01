//! # Lau Naturality Boundary
//!
//! The Naturality Boundary theorem: a computation is compile-time-eliminable
//! (zero per-instance cost) if and only if it is a natural transformation —
//! uniform in the instance, factoring through a universal property. The
//! irreducible runtime residue is bounded by Kolmogorov complexity K(answer | structure).
//!
//! You cannot push real information across the boundary.

pub mod natural_transformation;
pub mod classifier;
pub mod kolmogorov;
pub mod universal_property;
pub mod yoneda;
pub mod parametricity;
pub mod residue;
pub mod conservation;
pub mod analysis;
pub mod case_studies;

pub use natural_transformation::*;
pub use classifier::*;
pub use kolmogorov::*;
pub use universal_property::*;
pub use yoneda::*;
pub use parametricity::*;
pub use residue::*;
pub use conservation::*;
pub use analysis::*;
pub use case_studies::*;
