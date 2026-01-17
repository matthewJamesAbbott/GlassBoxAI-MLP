/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

pub const EPSILON: f64 = 1e-15;
pub const BLOCK_SIZE: u32 = 256;
pub const MODEL_MAGIC: &str = "MLPCUDA1";

mod kernels;
mod mlp;
mod cli;

fn main() {
    cli::run();
}
