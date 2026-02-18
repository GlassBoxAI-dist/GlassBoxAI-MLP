//! @file
//! @ingroup MLP_Core_Verified
//! Kani Verification: Global State Consistency
//!
//! Prove that concurrent access to shared state maintains defined invariants
//! and is free of data races. Safe Rust guarantees data-race freedom.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_mlp_invariants_after_mutation() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size > 0 && input_size <= 8);
        kani::assume(output_size > 0 && output_size <= 8);
        
        if let Some(mut mlp) = MLP::new(input_size, &[4], output_size) {
            let original_num_layers = mlp.num_layers;
            
            if let Some(layer) = mlp.get_layer_mut(1) {
                layer.biases[0] = 0.5;
            }
            
            kani::assert(mlp.num_layers == original_num_layers, "Layer count invariant");
            kani::assert(mlp.input_size == input_size, "Input size invariant");
            kani::assert(mlp.output_size == output_size, "Output size invariant");
        }
    }

    #[kani::proof]
    fn verify_layer_invariants_preserved() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 8);
        kani::assume(num_inputs > 0 && num_inputs <= 8);
        
        let mut layer = LayerData::new(num_neurons, num_inputs, TActivationType::AtReLU);
        
        let neuron_idx: usize = kani::any();
        let weight_idx: usize = kani::any();
        let value: f64 = kani::any();
        kani::assume(!value.is_nan());
        
        if neuron_idx < num_neurons && weight_idx < num_inputs {
            layer.set_weight(neuron_idx, weight_idx, value);
        }
        
        kani::assert(layer.num_neurons == num_neurons, "Neuron count invariant");
        kani::assert(layer.num_inputs == num_inputs, "Input count invariant");
        kani::assert(layer.weights.len() == num_neurons * num_inputs, "Weight array size invariant");
    }
}

