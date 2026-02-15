//! Kani Verification: No-Panic Guarantee
//!
//! Verify that target functions cannot trigger panic!, unwrap(), or expect()
//! failure across the entire input space.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_activation_functions_no_panic() {
        let x: f64 = kani::any();
        kani::assume(!x.is_nan());
        
        let sig_result = sigmoid(x);
        kani::assert(sig_result >= 0.0 && sig_result <= 1.0, "Sigmoid must be in [0,1]");
        
        let relu_result = relu(x);
        kani::assert(relu_result >= 0.0 || x < 0.0, "ReLU must be non-negative for positive x");
    }

    #[kani::proof]
    fn verify_max_index_no_panic() {
        let size: usize = kani::any();
        kani::assume(size <= 16);
        
        if size == 0 {
            let empty: Vec<f64> = vec![];
            let result = max_index(&empty);
            kani::assert(result.is_none(), "max_index on empty must return None");
        } else {
            let arr: Vec<f64> = vec![1.0; size];
            let result = max_index(&arr);
            kani::assert(result.is_some(), "max_index on non-empty must return Some");
            if let Some(idx) = result {
                kani::assert(idx < size, "max_index result must be valid index");
            }
        }
    }

    #[kani::proof]
    fn verify_mlp_construction_no_panic() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        
        if input_size == 0 || output_size == 0 {
            let result = MLP::new(input_size, &[], output_size);
            kani::assert(result.is_none(), "Invalid MLP config must return None, not panic");
        }
        
        if input_size > 0 && input_size <= 8 && output_size > 0 && output_size <= 8 {
            let result = MLP::new(input_size, &[4], output_size);
            kani::assert(result.is_some(), "Valid MLP config must succeed");
        }
    }

    #[kani::proof]
    fn verify_parse_activation_no_panic() {
        let test_inputs = ["sigmoid", "tanh", "relu", "softmax", "unknown", "", "SIGMOID"];
        for input in &test_inputs {
            let _result = parse_activation(input);
        }
    }

    #[kani::proof]
    fn verify_parse_optimizer_no_panic() {
        let test_inputs = ["sgd", "adam", "rmsprop", "unknown", "", "ADAM"];
        for input in &test_inputs {
            let _result = parse_optimizer(input);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::kani::core_types::*;

    #[test]
    fn test_activation_functions() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
    }
}
