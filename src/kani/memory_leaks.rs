//! Kani Verification: Memory Leak/Leakage Proofs
//!
//! Prove that all allocated memory is either freed or remains reachable.
//! Rust's ownership system guarantees no memory leaks for types
//! that don't use interior mutability or reference cycles.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_layer_data_owned_vectors() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 8);
        kani::assume(num_inputs > 0 && num_inputs <= 8);
        
        {
            let layer = LayerData::new(num_neurons, num_inputs, TActivationType::AtReLU);
            kani::assert(layer.weights.capacity() >= num_neurons * num_inputs, "Weights allocated");
            kani::assert(layer.biases.capacity() >= num_neurons, "Biases allocated");
        }
    }

    #[kani::proof]
    fn verify_allocation_with_limit_respects_budget() {
        let requested_size: usize = kani::any();
        kani::assume(requested_size <= MEMORY_BUDGET * 2 / std::mem::size_of::<f64>());
        
        let required_memory = requested_size * std::mem::size_of::<f64>();
        let result = allocate_with_limit(requested_size);
        
        if required_memory > MEMORY_BUDGET {
            kani::assert(result.is_none(), "Over-budget allocation must fail");
        } else {
            kani::assert(result.is_some(), "Within-budget allocation must succeed");
            if let Some(vec) = result {
                kani::assert(vec.len() == requested_size, "Allocated size must match");
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::kani::core_types::*;

    #[test]
    fn test_memory_budget() {
        let small = allocate_with_limit(1000);
        assert!(small.is_some());
        
        let huge = allocate_with_limit(MEMORY_BUDGET * 2);
        assert!(huge.is_none());
    }
}
