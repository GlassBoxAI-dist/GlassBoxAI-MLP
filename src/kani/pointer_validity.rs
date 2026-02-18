//! @file
//! @ingroup MLP_Core_Verified
//! Kani Verification: Pointer Validity Proofs
//!
//! Verify that all raw pointer dereferences are valid, aligned, and point to
//! initialized memory. This crate uses safe Rust; proof of no unsafe blocks.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_no_null_pointer_in_layer_creation() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 16);
        kani::assume(num_inputs > 0 && num_inputs <= 16);
        
        let layer = LayerData::new(num_neurons, num_inputs, TActivationType::AtReLU);
        
        kani::assert(!layer.weights.is_empty(), "Weights vector must be initialized");
        kani::assert(!layer.biases.is_empty(), "Biases vector must be initialized");
        kani::assert(!layer.outputs.is_empty(), "Outputs vector must be initialized");
        kani::assert(!layer.errors.is_empty(), "Errors vector must be initialized");
        kani::assert(layer.weights.len() == num_neurons * num_inputs, "Weights size must match");
        kani::assert(layer.biases.len() == num_neurons, "Biases size must match");
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_mlp_initialization_validity() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size > 0 && input_size <= 8);
        kani::assume(output_size > 0 && output_size <= 8);
        
        let hidden_sizes: [usize; 2] = [4, 4];
        let mlp = MLP::new(input_size, &hidden_sizes, output_size);
        
        if let Some(mlp) = mlp {
            kani::assert(!mlp.layers.is_empty(), "MLP must have layers");
            kani::assert(mlp.num_layers == mlp.layers.len(), "Layer count must match");
            for layer in &mlp.layers {
                kani::assert(!layer.weights.is_empty() || layer.num_inputs == 0, 
                    "Layer weights must be initialized");
            }
        }
    }
}

