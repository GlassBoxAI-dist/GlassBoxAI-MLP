//! @file
//! @ingroup MLP_Internal_Logic
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

#[cfg(feature = "cli")]
fn main() {
    facaded_mlp_cuda::cli::run();
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("Error: This binary requires the 'cli' feature to be enabled.");
    eprintln!("Build with: cargo build --features cli");
    std::process::exit(1);
}

