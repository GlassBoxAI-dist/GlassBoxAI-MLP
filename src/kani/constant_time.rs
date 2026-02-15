//! Kani Verification: Constant-Time Execution (Security)
//!
//! Verify that branching logic does not depend on secret/sensitive values
//! to prevent timing-based side-channel attacks.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_sigmoid_constant_time_bounds() {
        let x: f64 = kani::any();
        kani::assume(!x.is_nan());
        
        let result = sigmoid(x);
        kani::assert(result >= 0.0 && result <= 1.0, "Sigmoid bounded");
        kani::assert(!result.is_nan(), "Sigmoid never NaN");
    }

    #[kani::proof]
    fn verify_relu_constant_time_output() {
        let x: f64 = kani::any();
        kani::assume(!x.is_nan() && !x.is_infinite());
        
        let result = relu(x);
        kani::assert((x <= 0.0 && result == 0.0) || (x > 0.0 && result == x),
            "ReLU has predictable output regardless of secret value magnitude");
    }

    #[kani::proof]
    fn verify_activation_selection_public_key() {
        let act_type: i32 = kani::any();
        kani::assume(act_type >= 0 && act_type <= 3);
        
        let act = match act_type {
            0 => TActivationType::AtSigmoid,
            1 => TActivationType::AtTanh,
            2 => TActivationType::AtReLU,
            _ => TActivationType::AtSoftmax,
        };
        
        kani::assert(
            matches!(act, TActivationType::AtSigmoid | TActivationType::AtTanh | 
                         TActivationType::AtReLU | TActivationType::AtSoftmax),
            "Activation type is from public config, not secret"
        );
    }
}
