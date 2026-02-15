//! Kani Verification: Strict Bound Checks
//!
//! Prove that all collection indexing is mathematically incapable of
//! out-of-bounds access under any symbolic input.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_array_bounds_layer_access() {
        let num_layers: usize = kani::any();
        kani::assume(num_layers > 0 && num_layers <= MAX_LAYERS);
        
        let layer_idx: usize = kani::any();
        
        let mlp = MLP::new(4, &[8, 8], 2);
        if let Some(mlp) = mlp {
            let result = mlp.get_layer(layer_idx);
            if layer_idx >= mlp.num_layers {
                kani::assert(result.is_none(), "Out-of-bounds layer access must return None");
            }
        }
    }

    #[kani::proof]
    #[kani::unwind(20)]
    fn verify_array_bounds_weight_access() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 16);
        kani::assume(num_inputs > 0 && num_inputs <= 16);
        
        let layer = LayerData::new(num_neurons, num_inputs, TActivationType::AtReLU);
        
        let neuron_idx: usize = kani::any();
        let weight_idx: usize = kani::any();
        
        let result = layer.get_weight(neuron_idx, weight_idx);
        
        if neuron_idx >= num_neurons || weight_idx >= num_inputs {
            kani::assert(result.is_none(), "Out-of-bounds weight access must return None");
        } else {
            kani::assert(result.is_some(), "Valid weight access must return Some");
        }
    }

    #[kani::proof]
    #[kani::unwind(20)]
    fn verify_array_bounds_bias_access() {
        let num_neurons: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 16);
        
        let layer = LayerData::new(num_neurons, 4, TActivationType::AtSigmoid);
        
        let neuron_idx: usize = kani::any();
        let result = layer.get_bias(neuron_idx);
        
        if neuron_idx >= num_neurons {
            kani::assert(result.is_none(), "Out-of-bounds bias access must return None");
        } else {
            kani::assert(result.is_some(), "Valid bias access must return Some");
        }
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_array_bounds_output_access() {
        let num_neurons: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 16);
        
        let layer = LayerData::new(num_neurons, 4, TActivationType::AtSigmoid);
        
        let neuron_idx: usize = kani::any();
        let result = layer.get_output(neuron_idx);
        
        if neuron_idx >= num_neurons {
            kani::assert(result.is_none(), "Out-of-bounds output access must return None");
        } else {
            kani::assert(result.is_some(), "Valid output access must return Some");
        }
    }

    #[kani::proof]
    #[kani::unwind(32)]
    fn verify_validate_bounds_generic() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 32);
        
        let arr: Vec<f64> = vec![0.0; size];
        let idx: usize = kani::any();
        
        let result = validate_bounds(&arr, idx);
        
        if idx >= size {
            kani::assert(result.is_none(), "validate_bounds must return None for out-of-bounds");
        } else {
            kani::assert(result.is_some(), "validate_bounds must return Some for valid index");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::kani::core_types::*;

    #[test]
    fn test_layer_bounds() {
        let layer = LayerData::new(8, 4, TActivationType::AtReLU);
        assert!(layer.get_weight(0, 0).is_some());
        assert!(layer.get_weight(7, 3).is_some());
        assert!(layer.get_weight(8, 0).is_none());
        assert!(layer.get_weight(0, 4).is_none());
    }
}
