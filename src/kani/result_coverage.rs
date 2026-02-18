//! @file
//! @ingroup MLP_Core_Verified
//! Kani Verification: Result Coverage Audit
//!
//! Verify that all Error variants in returned Result types are explicitly
//! handled and do not leave the system in an indeterminate state.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_layer_access_result_handling() {
        let num_neurons: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 8);
        
        let layer = LayerData::new(num_neurons, 4, TActivationType::AtSigmoid);
        let idx: usize = kani::any();
        
        match layer.get_bias(idx) {
            Some(bias) => {
                kani::assert(idx < num_neurons, "Some implies valid index");
                kani::assert(!bias.is_nan(), "Bias must not be NaN");
            }
            None => {
                kani::assert(idx >= num_neurons, "None implies invalid index");
            }
        }
    }

    #[kani::proof]
    fn verify_mlp_creation_result_handling() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size <= 16 && output_size <= 16);
        
        match MLP::new(input_size, &[4], output_size) {
            Some(mlp) => {
                kani::assert(input_size > 0 && output_size > 0, "Some implies valid sizes");
                kani::assert(mlp.num_layers >= 2, "MLP must have at least 2 layers");
            }
            None => {
                kani::assert(input_size == 0 || output_size == 0, "None implies invalid sizes");
            }
        }
    }

    #[kani::proof]
    fn verify_compute_loss_result_handling() {
        let size: usize = kani::any();
        kani::assume(size <= 8);
        
        let predicted = vec![0.5; size];
        let target = vec![1.0; size];
        
        match compute_loss_checked(&predicted, &target, false) {
            Some(loss) => {
                kani::assert(size > 0, "Some implies non-empty arrays");
                kani::assert(is_fp_sane(loss), "Loss must be finite");
                kani::assert(loss >= 0.0, "Loss must be non-negative");
            }
            None => {
                kani::assert(size == 0, "None implies empty arrays");
            }
        }
    }
}

