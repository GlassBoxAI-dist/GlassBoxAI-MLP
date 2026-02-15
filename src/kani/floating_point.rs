//! Kani Verification: Floating-Point Sanity
//!
//! Prove that operations involving f32/f64 never result in unhandled NaN or
//! Infinity states that could bypass logic checks.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_fp_sanity_check() {
        let value: f64 = kani::any();
        
        let is_sane = is_fp_sane(value);
        
        if value.is_nan() || value.is_infinite() {
            kani::assert(!is_sane, "NaN/Infinity must be flagged as not sane");
        } else {
            kani::assert(is_sane, "Finite values must be flagged as sane");
        }
    }

    #[kani::proof]
    fn verify_clamp_fp_handles_special_values() {
        let value: f64 = kani::any();
        let min: f64 = 0.0;
        let max: f64 = 1.0;
        
        let result = clamp_fp(value, min, max);
        
        if value.is_nan() || value.is_infinite() {
            kani::assert(result.is_none(), "Special values must return None");
        } else {
            kani::assert(result.is_some(), "Normal values must return Some");
            if let Some(clamped) = result {
                kani::assert(clamped >= min && clamped <= max, "Clamped value in range");
            }
        }
    }

    #[kani::proof]
    fn verify_sigmoid_never_nan_or_inf() {
        let x: f64 = kani::any();
        kani::assume(!x.is_nan());
        
        let result = sigmoid(x);
        
        kani::assert(!result.is_nan(), "Sigmoid never produces NaN");
        kani::assert(!result.is_infinite(), "Sigmoid never produces Infinity");
        kani::assert(result >= 0.0 && result <= 1.0, "Sigmoid always in [0,1]");
    }

    #[kani::proof]
    fn verify_relu_never_nan() {
        let x: f64 = kani::any();
        kani::assume(!x.is_nan());
        
        let result = relu(x);
        
        kani::assert(!result.is_nan(), "ReLU never produces NaN");
        if x.is_finite() {
            kani::assert(!result.is_infinite() || x.is_infinite(), "ReLU preserves finiteness");
        }
    }

    #[kani::proof]
    fn verify_compute_loss_nan_handling() {
        let size: usize = 4;
        let mut predicted = vec![0.5; size];
        let target = vec![1.0; size];
        
        let val: f64 = kani::any();
        kani::assume(val.is_nan());
        predicted[0] = val;
        
        let result = compute_loss_checked(&predicted, &target, false);
        kani::assert(result.is_none(), "NaN in input must cause None result");
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::kani::core_types::*;

    #[test]
    fn test_fp_sanity() {
        assert!(is_fp_sane(1.0));
        assert!(is_fp_sane(0.0));
        assert!(!is_fp_sane(f64::NAN));
        assert!(!is_fp_sane(f64::INFINITY));
    }
}
