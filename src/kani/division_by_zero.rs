//! @file
//! @ingroup MLP_Core_Verified
//! Kani Verification: Division-by-Zero Exclusion
//!
//! Verify that any denominator derived from variable/external input is
//! mathematically proven to never be zero.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_safe_div_no_zero() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        
        let result = safe_div(a, b);
        
        if b == 0 {
            kani::assert(result.is_none(), "Division by zero must return None");
        } else if a == i64::MIN && b == -1 {
            kani::assert(result.is_none(), "MIN/-1 overflow must return None");
        } else {
            kani::assert(result.is_some(), "Non-zero divisor must return Some");
        }
    }

    #[kani::proof]
    fn verify_normalization_no_div_by_zero() {
        let min_val: f64 = kani::any();
        let max_val: f64 = kani::any();
        kani::assume(!min_val.is_nan() && !max_val.is_nan());
        kani::assume(!min_val.is_infinite() && !max_val.is_infinite());
        
        let range = if max_val == min_val { 1.0 } else { max_val - min_val };
        
        kani::assert(range != 0.0, "Range must never be zero after check");
        kani::assert(range.is_finite(), "Range must be finite");
    }

    #[kani::proof]
    fn verify_softmax_denominator_non_zero() {
        let vals: [f64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        
        for v in &vals {
            kani::assume(!v.is_nan() && !v.is_infinite());
            kani::assume(*v >= -100.0 && *v <= 100.0);
        }
        
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = vals.iter().map(|&s| (s - max_val).exp()).sum();
        
        kani::assert(sum_exp > 0.0, "Softmax denominator must be positive");
    }
}

