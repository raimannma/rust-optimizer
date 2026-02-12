#![allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

mod bohb;
#[cfg(feature = "cma-es")]
mod cma_es;
mod differential_evolution;
#[cfg(feature = "gp")]
mod gp;
mod multivariate_tpe;
mod random;
mod tpe;
