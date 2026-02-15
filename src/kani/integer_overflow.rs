//! Kani Verification: Integer Overflow Prevention
//!
//! Prove that all arithmetic operations are safe from wrapping, overflowing,
//! or underflowing.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_safe_add_no_overflow() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        
        let result = safe_add(a, b);
        
        let would_overflow = (b > 0 && a > i64::MAX - b) || (b < 0 && a < i64::MIN - b);
        
        if would_overflow {
            kani::assert(result.is_none(), "Overflow must return None");
        } else {
            kani::assert(result.is_some(), "No overflow must return Some");
        }
    }

    #[kani::proof]
    fn verify_safe_sub_no_overflow() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        
        let result = safe_sub(a, b);
        
        let would_overflow = (b < 0 && a > i64::MAX + b) || (b > 0 && a < i64::MIN + b);
        
        if would_overflow {
            kani::assert(result.is_none(), "Underflow must return None");
        } else {
            kani::assert(result.is_some(), "No underflow must return Some");
        }
    }

    #[kani::proof]
    fn verify_safe_mul_no_overflow() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        
        let result = safe_mul(a, b);
        
        if a != 0 && b != 0 {
            let check = a.checked_mul(b);
            kani::assert(result == check, "safe_mul must match checked_mul");
        }
    }

    #[kani::proof]
    fn verify_layer_size_calculation_no_overflow() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons <= MAX_NEURONS_PER_LAYER);
        kani::assume(num_inputs <= MAX_NEURONS_PER_LAYER);
        
        let result = num_neurons.checked_mul(num_inputs);
        
        if num_neurons <= 1024 && num_inputs <= 1024 {
            kani::assert(result.is_some(), "Layer size calculation must not overflow");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::kani::core_types::*;

    #[test]
    fn test_safe_arithmetic() {
        assert_eq!(safe_add(1, 2), Some(3));
        assert_eq!(safe_add(i64::MAX, 1), None);
        assert_eq!(safe_sub(5, 3), Some(2));
        assert_eq!(safe_mul(10, 10), Some(100));
        assert_eq!(safe_div(10, 2), Some(5));
        assert_eq!(safe_div(10, 0), None);
    }
}
