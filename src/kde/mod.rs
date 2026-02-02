//! Kernel Density Estimation for parameter distributions.
//!
//! This module provides kernel density estimators used by TPE samplers
//! to model probability distributions over good and bad trial regions.
//!
//! - [`univariate`] - Univariate (single-parameter) KDE
//! - [`multivariate`] - Multivariate (joint-parameter) KDE for capturing parameter dependencies

mod multivariate;
mod univariate;

pub(crate) use multivariate::MultivariateKDE;
pub(crate) use univariate::KernelDensityEstimator;
