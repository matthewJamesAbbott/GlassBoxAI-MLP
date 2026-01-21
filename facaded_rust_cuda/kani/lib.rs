//! Kani Verification Test Suite for Facaded MLP CUDA
//! 
//! This module provides formal verification harnesses following CISA "Secure by Design"
//! standards. Each harness uses symbolic inputs to mathematically prove safety properties.
//!
//! Run with: `cargo kani --tests`

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub mod core_types;
pub mod harnesses;

pub use core_types::*;
