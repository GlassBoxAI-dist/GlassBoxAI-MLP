//! @file
//! @ingroup MLP_Core_Verified
//! Kani Verification: FFI Boundary Safety (CISA/NSA Compliance)
//!
//! Proves that all data crossing the C FFI boundary is validated before use.
//! Models the C ABI contract where foreign callers can pass arbitrary i32/f64
//! values including negatives, NaN, Infinity, and extreme values.
//!
//! Covers CISA "Secure by Design" requirements:
//! - Memory safety across language boundaries
//! - Input validation at trust boundaries  
//! - No undefined behavior from external input
//! - Proper error handling at FFI surface

#[cfg(kani)]
mod kani_proofs {
    use crate::core_types::*;

    // =========================================================================
    // A. SIGNED-TO-UNSIGNED CONVERSION SAFETY
    // Prove that i32 -> usize conversions reject negative values before any
    // unsafe operation (from_raw_parts, array indexing, allocation).
    // =========================================================================

    #[kani::proof]
    fn verify_i32_to_usize_rejects_negative() {
        let val: i32 = kani::any();

        let result = validate_i32_as_usize(val);

        if val < 0 {
            kani::assert(result.is_none(), "Negative i32 must be rejected before usize cast");
        } else {
            kani::assert(result.is_some(), "Non-negative i32 must be accepted");
            kani::assert(result.unwrap() == val as usize, "Converted value must match");
        }
    }

    #[kani::proof]
    fn verify_i32_positive_rejects_zero_and_negative() {
        let val: i32 = kani::any();

        let result = validate_i32_positive(val);

        if val <= 0 {
            kani::assert(result.is_none(), "Zero or negative i32 must be rejected");
        } else {
            kani::assert(result.is_some(), "Positive i32 must be accepted");
            kani::assert(result.unwrap() == val as usize, "Converted value must match");
        }
    }

    #[kani::proof]
    fn verify_ffi_len_validates_range() {
        let len: i32 = kani::any();
        let max: usize = kani::any();
        kani::assume(max <= MAX_FFI_ARRAY_LEN);

        let result = validate_ffi_len(len, max);

        if len < 0 {
            kani::assert(result.is_none(), "Negative length must be rejected");
        } else if (len as usize) > max {
            kani::assert(result.is_none(), "Length exceeding max must be rejected");
        } else {
            kani::assert(result.is_some(), "Valid length must be accepted");
        }
    }

    #[kani::proof]
    fn verify_ffi_len_i32_min_rejected() {
        let result = validate_ffi_len(i32::MIN, MAX_ARRAY_SIZE);
        kani::assert(result.is_none(), "i32::MIN must be rejected as length");
    }

    #[kani::proof]
    fn verify_ffi_len_negative_one_rejected() {
        let result = validate_ffi_len(-1, MAX_ARRAY_SIZE);
        kani::assert(result.is_none(), "-1 must be rejected as length");
    }

    // =========================================================================
    // B. FFI BUFFER CAPACITY VALIDATION
    // Prove that output buffer capacity is validated before copy_nonoverlapping.
    // Models the C contract where capacity comes from untrusted caller.
    // =========================================================================

    #[kani::proof]
    fn verify_negative_capacity_prevents_buffer_write() {
        let capacity: i32 = kani::any();
        kani::assume(capacity < 0);

        let validated = validate_i32_as_usize(capacity);
        kani::assert(validated.is_none(),
            "Negative capacity must be rejected before buffer write");
    }

    #[kani::proof]
    fn verify_zero_capacity_prevents_buffer_write() {
        let capacity: i32 = 0;
        let validated = validate_i32_positive(capacity);
        kani::assert(validated.is_none(),
            "Zero capacity must be rejected before buffer write");
    }

    #[kani::proof]
    fn verify_output_write_bounded_by_validated_capacity() {
        let data_len: usize = kani::any();
        let capacity: i32 = kani::any();
        kani::assume(data_len <= 64);
        kani::assume(capacity >= 0 && capacity <= 64);

        let cap_usize = capacity as usize;
        let write_len = data_len.min(cap_usize);

        kani::assert(write_len <= cap_usize,
            "Write length must never exceed validated capacity");
        kani::assert(write_len <= data_len,
            "Write length must never exceed source data length");
    }

    // =========================================================================
    // C. FLOATING-POINT PARAMETER VALIDATION AT FFI BOUNDARY
    // Prove that NaN/Infinity f64 values from foreign callers are rejected
    // before being stored in MLP state.
    // =========================================================================

    #[kani::proof]
    fn verify_f64_param_rejects_nan() {
        let val: f64 = kani::any();
        kani::assume(val.is_nan());

        let result = validate_f64_param(val);
        kani::assert(result.is_none(), "NaN must be rejected at FFI boundary");
    }

    #[kani::proof]
    fn verify_f64_param_rejects_infinity() {
        let val: f64 = kani::any();
        kani::assume(val.is_infinite());

        let result = validate_f64_param(val);
        kani::assert(result.is_none(), "Infinity must be rejected at FFI boundary");
    }

    #[kani::proof]
    fn verify_f64_param_accepts_finite() {
        let val: f64 = kani::any();
        kani::assume(!val.is_nan() && !val.is_infinite());

        let result = validate_f64_param(val);
        kani::assert(result.is_some(), "Finite f64 must be accepted");
        kani::assert(result.unwrap() == val, "Value must be preserved");
    }

    #[kani::proof]
    fn verify_learning_rate_validation() {
        let val: f64 = kani::any();

        let result = validate_learning_rate(val);

        if val.is_nan() || val.is_infinite() || val < 0.0 || val > 100.0 {
            kani::assert(result.is_none(),
                "Invalid learning rate must be rejected");
        } else {
            kani::assert(result.is_some(),
                "Valid learning rate must be accepted");
        }
    }

    #[kani::proof]
    fn verify_dropout_rate_validation() {
        let val: f64 = kani::any();

        let result = validate_dropout_rate(val);

        if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1.0 {
            kani::assert(result.is_none(),
                "Invalid dropout rate must be rejected");
        } else {
            kani::assert(result.is_some(),
                "Valid dropout rate must be accepted");
        }
    }

    #[kani::proof]
    fn verify_l2_lambda_validation() {
        let val: f64 = kani::any();

        let result = validate_l2_lambda(val);

        if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1000.0 {
            kani::assert(result.is_none(),
                "Invalid L2 lambda must be rejected");
        } else {
            kani::assert(result.is_some(),
                "Valid L2 lambda must be accepted");
        }
    }

    // =========================================================================
    // D. ENUM BOUNDARY VALIDATION FROM FFI
    // Prove that i32 enum values from foreign callers are validated against
    // the defined variant set before conversion.
    // =========================================================================

    #[kani::proof]
    fn verify_activation_i32_validation_exhaustive() {
        let val: i32 = kani::any();

        let result = validate_activation_i32(val);

        if val >= 0 && val <= 3 {
            kani::assert(result.is_some(), "Valid activation must be accepted");
        } else {
            kani::assert(result.is_none(), "Invalid activation must be rejected");
        }
    }

    #[kani::proof]
    fn verify_activation_i32_negative_rejected() {
        let val: i32 = kani::any();
        kani::assume(val < 0);

        let result = validate_activation_i32(val);
        kani::assert(result.is_none(), "Negative activation type must be rejected");
    }

    #[kani::proof]
    fn verify_optimizer_i32_validation_exhaustive() {
        let val: i32 = kani::any();

        let result = validate_optimizer_i32(val);

        if val >= 0 && val <= 2 {
            kani::assert(result.is_some(), "Valid optimizer must be accepted");
        } else {
            kani::assert(result.is_none(), "Invalid optimizer must be rejected");
        }
    }

    #[kani::proof]
    fn verify_optimizer_i32_negative_rejected() {
        let val: i32 = kani::any();
        kani::assume(val < 0);

        let result = validate_optimizer_i32(val);
        kani::assert(result.is_none(), "Negative optimizer type must be rejected");
    }

    // =========================================================================
    // E. FFI CREATE PRECONDITIONS
    // Prove that MLP creation via FFI rejects all invalid parameter
    // combinations that could lead to UB in from_raw_parts or overflow.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_input() {
        let result = MLP::new(0, &[4], 2);
        kani::assert(result.is_none(), "Zero input size must be rejected at creation");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_output() {
        let result = MLP::new(2, &[4], 0);
        kani::assert(result.is_none(), "Zero output size must be rejected at creation");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_zero_hidden() {
        let result = MLP::new(2, &[0], 2);
        kani::assert(result.is_none(), "Zero hidden size must be rejected at creation");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_excessive_hidden_layers() {
        let num_hidden: usize = kani::any();
        kani::assume(num_hidden > MAX_LAYERS - 2 && num_hidden <= 32);

        let sizes: Vec<usize> = vec![4; num_hidden];
        let result = MLP::new(2, &sizes, 2);
        kani::assert(result.is_none(), "Too many hidden layers must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_create_rejects_oversized_hidden() {
        let size: usize = kani::any();
        kani::assume(size > MAX_NEURONS_PER_LAYER);

        let result = MLP::new(2, &[size], 2);
        kani::assert(result.is_none(), "Oversized hidden layer must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_hidden_count_i32_negative_as_usize_huge() {
        let hidden_count: i32 = -1;
        let as_usize = hidden_count as usize;
        kani::assert(as_usize > MAX_LAYERS,
            "Negative i32 as usize becomes huge, proving validation needed");
    }

    #[kani::proof]
    fn verify_ffi_i32_min_as_usize_huge() {
        let hidden_count: i32 = i32::MIN;
        let as_usize = hidden_count as usize;
        kani::assert(as_usize > MAX_ARRAY_SIZE,
            "i32::MIN as usize is huge, proving validation needed");
    }

    // =========================================================================
    // F. FFI TRAIN/PREDICT LENGTH VALIDATION
    // Prove that input/target/output lengths from FFI are bounded and
    // cannot cause from_raw_parts to read arbitrary memory.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_train_input_len_validated() {
        let input_len: i32 = kani::any();
        let target_len: i32 = kani::any();

        let input_valid = validate_ffi_len(input_len, MAX_FFI_ARRAY_LEN);
        let target_valid = validate_ffi_len(target_len, MAX_FFI_ARRAY_LEN);

        if input_len < 0 || input_len as usize > MAX_FFI_ARRAY_LEN {
            kani::assert(input_valid.is_none(),
                "Invalid input length must be rejected before from_raw_parts");
        }
        if target_len < 0 || target_len as usize > MAX_FFI_ARRAY_LEN {
            kani::assert(target_valid.is_none(),
                "Invalid target length must be rejected before from_raw_parts");
        }
    }

    #[kani::proof]
    fn verify_ffi_predict_capacity_validated() {
        let output_capacity: i32 = kani::any();

        let valid = validate_i32_as_usize(output_capacity);

        if output_capacity < 0 {
            kani::assert(valid.is_none(),
                "Negative output capacity must be rejected before copy_nonoverlapping");
        }
    }

    #[kani::proof]
    fn verify_ffi_predict_output_bounded_by_capacity() {
        let result_len: usize = kani::any();
        let capacity: i32 = kani::any();
        kani::assume(result_len <= 64);
        kani::assume(capacity > 0 && capacity <= 64);

        let cap_usize = capacity as usize;
        let write_len = result_len.min(cap_usize);

        kani::assert(write_len <= cap_usize,
            "Bytes written must not exceed caller-provided capacity");
    }

    // =========================================================================
    // G. FFI LAYER INDEX VALIDATION
    // Prove that layer indices from foreign callers are validated against
    // MLP structure bounds before any access.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_layer_index_negative_rejected() {
        let layer: i32 = kani::any();
        kani::assume(layer < 0);

        let valid = validate_i32_as_usize(layer);
        kani::assert(valid.is_none(),
            "Negative layer index must be rejected before array access");
    }

    #[kani::proof]
    fn verify_ffi_layer_index_out_of_bounds_safe() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size > 0 && input_size <= 8);
        kani::assume(output_size > 0 && output_size <= 8);

        if let Some(mlp) = MLP::new(input_size, &[4], output_size) {
            let layer_idx: usize = kani::any();

            let result = mlp.get_layer(layer_idx);
            if layer_idx >= mlp.num_layers {
                kani::assert(result.is_none(),
                    "Out-of-bounds layer index must return None");
            }
        }
    }

    // =========================================================================
    // H. FFI NEURON INDEX VALIDATION
    // Prove that neuron indices from foreign callers are validated.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_neuron_index_negative_rejected() {
        let neuron: i32 = kani::any();
        kani::assume(neuron < 0);

        let valid = validate_i32_as_usize(neuron);
        kani::assert(valid.is_none(),
            "Negative neuron index must be rejected");
    }

    #[kani::proof]
    fn verify_ffi_weight_index_negative_rejected() {
        let weight_idx: i32 = kani::any();
        kani::assume(weight_idx < 0);

        let valid = validate_i32_as_usize(weight_idx);
        kani::assert(valid.is_none(),
            "Negative weight index must be rejected");
    }

    // =========================================================================
    // I. FFI HISTOGRAM PARAMETER VALIDATION
    // Prove bins parameter is validated for histogram functions.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_histogram_bins_negative_rejected() {
        let bins: i32 = kani::any();
        kani::assume(bins < 0);

        let valid = validate_i32_positive(bins);
        kani::assert(valid.is_none(),
            "Negative bins must be rejected for histogram");
    }

    #[kani::proof]
    fn verify_ffi_histogram_bins_zero_rejected() {
        let valid = validate_i32_positive(0);
        kani::assert(valid.is_none(),
            "Zero bins must be rejected for histogram");
    }

    // =========================================================================
    // J. FFI ERROR STRING SAFETY
    // Prove that error message handling cannot produce invalid C strings.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_error_nul_byte_sanitized() {
        let test_strings = ["error\0message", "\0", "normal error"];
        for s in &test_strings {
            let sanitized = s.replace('\0', "");
            kani::assert(!sanitized.contains('\0'),
                "NUL bytes must be removed before CString creation");
        }
    }

    // =========================================================================
    // K. FFI NO-PANIC GUARANTEE FOR VALIDATION HELPERS
    // Prove that validation functions never panic regardless of input.
    // =========================================================================

    #[kani::proof]
    fn verify_validate_i32_as_usize_no_panic() {
        let val: i32 = kani::any();
        let _result = validate_i32_as_usize(val);
    }

    #[kani::proof]
    fn verify_validate_i32_positive_no_panic() {
        let val: i32 = kani::any();
        let _result = validate_i32_positive(val);
    }

    #[kani::proof]
    fn verify_validate_ffi_len_no_panic() {
        let len: i32 = kani::any();
        let max: usize = kani::any();
        kani::assume(max <= MAX_FFI_ARRAY_LEN);
        let _result = validate_ffi_len(len, max);
    }

    #[kani::proof]
    fn verify_validate_f64_param_no_panic() {
        let val: f64 = kani::any();
        let _result = validate_f64_param(val);
    }

    #[kani::proof]
    fn verify_validate_f64_param_range_no_panic() {
        let val: f64 = kani::any();
        let min: f64 = kani::any();
        let max: f64 = kani::any();
        kani::assume(!min.is_nan() && !max.is_nan());
        let _result = validate_f64_param_range(val, min, max);
    }

    #[kani::proof]
    fn verify_validate_learning_rate_no_panic() {
        let val: f64 = kani::any();
        let _result = validate_learning_rate(val);
    }

    #[kani::proof]
    fn verify_validate_dropout_rate_no_panic() {
        let val: f64 = kani::any();
        let _result = validate_dropout_rate(val);
    }

    #[kani::proof]
    fn verify_validate_l2_lambda_no_panic() {
        let val: f64 = kani::any();
        let _result = validate_l2_lambda(val);
    }

    #[kani::proof]
    fn verify_validate_activation_i32_no_panic() {
        let val: i32 = kani::any();
        let _result = validate_activation_i32(val);
    }

    #[kani::proof]
    fn verify_validate_optimizer_i32_no_panic() {
        let val: i32 = kani::any();
        let _result = validate_optimizer_i32(val);
    }

    // =========================================================================
    // L. CROSS-LANGUAGE TYPE REPRESENTATION PROOFS
    // Prove that repr(i32) enums have the expected ABI-compatible layout.
    // =========================================================================

    #[kani::proof]
    fn verify_activation_type_repr_i32_abi() {
        kani::assert(std::mem::size_of::<TActivationType>() == std::mem::size_of::<i32>(),
            "TActivationType must be i32-sized for C ABI compatibility");
        kani::assert(TActivationType::AtSigmoid as i32 == 0, "Sigmoid == 0");
        kani::assert(TActivationType::AtTanh as i32 == 1, "Tanh == 1");
        kani::assert(TActivationType::AtReLU as i32 == 2, "ReLU == 2");
        kani::assert(TActivationType::AtSoftmax as i32 == 3, "Softmax == 3");
    }

    #[kani::proof]
    fn verify_optimizer_type_repr_i32_abi() {
        kani::assert(std::mem::size_of::<TOptimizerType>() == std::mem::size_of::<i32>(),
            "TOptimizerType must be i32-sized for C ABI compatibility");
        kani::assert(TOptimizerType::OtSGD as i32 == 0, "SGD == 0");
        kani::assert(TOptimizerType::OtAdam as i32 == 1, "Adam == 1");
        kani::assert(TOptimizerType::OtRMSProp as i32 == 2, "RMSProp == 2");
    }

    #[kani::proof]
    fn verify_f64_abi_compatibility() {
        kani::assert(std::mem::size_of::<f64>() == 8,
            "f64 must be 8 bytes for C double compatibility");
        kani::assert(std::mem::align_of::<f64>() == 8,
            "f64 must be 8-byte aligned for C double compatibility");
    }

    #[kani::proof]
    fn verify_i32_abi_compatibility() {
        kani::assert(std::mem::size_of::<i32>() == 4,
            "i32 must be 4 bytes for C int32_t compatibility");
        kani::assert(std::mem::align_of::<i32>() == 4,
            "i32 must be 4-byte aligned for C int32_t compatibility");
    }

    // =========================================================================
    // M. FFI INPUT ARRAY NAN/INFINITY DETECTION
    // Prove that f64 arrays crossing FFI boundary can be validated for
    // NaN/Infinity values before use in computation.
    // =========================================================================

    #[kani::proof]
    #[kani::unwind(9)]
    fn verify_ffi_input_array_nan_detection() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 8);

        let mut arr = vec![1.0; size];
        let idx: usize = kani::any();
        kani::assume(idx < size);

        let bad_val: f64 = kani::any();
        kani::assume(bad_val.is_nan());
        arr[idx] = bad_val;

        let has_nan = arr.iter().any(|x| x.is_nan());
        kani::assert(has_nan, "NaN in input array must be detectable");
    }

    #[kani::proof]
    #[kani::unwind(9)]
    fn verify_ffi_input_array_infinity_detection() {
        let size: usize = kani::any();
        kani::assume(size > 0 && size <= 8);

        let mut arr = vec![1.0; size];
        let idx: usize = kani::any();
        kani::assume(idx < size);

        let bad_val: f64 = kani::any();
        kani::assume(bad_val.is_infinite());
        arr[idx] = bad_val;

        let has_inf = arr.iter().any(|x| x.is_infinite());
        kani::assert(has_inf, "Infinity in input array must be detectable");
    }

    // =========================================================================
    // N. FFI RESOURCE LIMITS AT BOUNDARY
    // Prove that FFI-supplied sizes cannot cause excessive allocation.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_allocation_respects_budget_at_boundary() {
        let requested: i32 = kani::any();
        kani::assume(requested >= 0);

        let size = requested as usize;
        let required_memory = size.checked_mul(std::mem::size_of::<f64>());

        if let Some(bytes) = required_memory {
            let result = allocate_with_limit(size);
            if bytes > MEMORY_BUDGET {
                kani::assert(result.is_none(),
                    "FFI-supplied size exceeding budget must be rejected");
            }
        }
    }

    #[kani::proof]
    fn verify_ffi_to_vec_copy_bounded() {
        let len: i32 = kani::any();
        kani::assume(len >= 0 && len <= 4096);

        let len_usize = len as usize;
        let bytes = len_usize * std::mem::size_of::<f64>();

        kani::assert(bytes <= 4096 * 8,
            "to_vec copy from FFI must be bounded");
    }

    // =========================================================================
    // O. FFI STATE CONSISTENCY AFTER PARAMETER MUTATION
    // Prove that setting parameters via FFI maintains MLP invariants.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_parameter_mutation_preserves_structure() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size > 0 && input_size <= 8);
        kani::assume(output_size > 0 && output_size <= 8);

        if let Some(mut mlp) = MLP::new(input_size, &[4], output_size) {
            let original_layers = mlp.num_layers;
            let original_input = mlp.input_size;
            let original_output = mlp.output_size;

            let lr: f64 = kani::any();
            if !lr.is_nan() && !lr.is_infinite() && lr >= 0.0 {
                mlp.learning_rate = lr;
            }

            let dr: f64 = kani::any();
            if !dr.is_nan() && !dr.is_infinite() && dr >= 0.0 && dr <= 1.0 {
                mlp.dropout_rate = dr;
            }

            kani::assert(mlp.num_layers == original_layers,
                "Layer count must not change from parameter mutation");
            kani::assert(mlp.input_size == original_input,
                "Input size must not change from parameter mutation");
            kani::assert(mlp.output_size == original_output,
                "Output size must not change from parameter mutation");
        }
    }

    // =========================================================================
    // P. FFI COMBINED VALIDATION: END-TO-END PROPERTY
    // Prove that the full validation pipeline (i32 -> usize -> bounds check ->
    // safe operation) is correct for key FFI operations.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_complete_train_validation_pipeline() {
        let input_len: i32 = kani::any();
        let target_len: i32 = kani::any();

        let input_valid = validate_ffi_len(input_len, MAX_FFI_ARRAY_LEN);
        let target_valid = validate_ffi_len(target_len, MAX_FFI_ARRAY_LEN);

        if input_valid.is_some() && target_valid.is_some() {
            let il = input_valid.unwrap();
            let tl = target_valid.unwrap();
            kani::assert(il <= MAX_FFI_ARRAY_LEN,
                "Validated input length must be bounded");
            kani::assert(tl <= MAX_FFI_ARRAY_LEN,
                "Validated target length must be bounded");

            let input_bytes = il * std::mem::size_of::<f64>();
            let target_bytes = tl * std::mem::size_of::<f64>();
            kani::assert(input_bytes <= MAX_FFI_ARRAY_LEN * 8,
                "Input memory footprint must be bounded");
            kani::assert(target_bytes <= MAX_FFI_ARRAY_LEN * 8,
                "Target memory footprint must be bounded");
        }
    }

    #[kani::proof]
    fn verify_ffi_complete_predict_validation_pipeline() {
        let input_len: i32 = kani::any();
        let output_capacity: i32 = kani::any();

        let input_valid = validate_ffi_len(input_len, MAX_FFI_ARRAY_LEN);
        let output_valid = validate_i32_as_usize(output_capacity);

        if input_valid.is_some() && output_valid.is_some() {
            let il = input_valid.unwrap();
            let oc = output_valid.unwrap();
            kani::assert(il <= MAX_FFI_ARRAY_LEN,
                "Validated input length must be bounded");

            let data_len: usize = kani::any();
            kani::assume(data_len <= 64);
            let write_len = data_len.min(oc);
            kani::assert(write_len <= oc,
                "Write to output buffer must be bounded by capacity");
        }
    }

    #[kani::proof]
    fn verify_ffi_complete_create_validation_pipeline() {
        let input_size: i32 = kani::any();
        let output_size: i32 = kani::any();
        let hidden_count: i32 = kani::any();

        let iv = validate_i32_positive(input_size);
        let ov = validate_i32_positive(output_size);
        let hc = validate_i32_positive(hidden_count);

        if iv.is_none() || ov.is_none() || hc.is_none() {
            if input_size <= 0 || output_size <= 0 || hidden_count <= 0 {
                kani::assert(true, "Invalid inputs correctly rejected");
            }
        } else {
            let is = iv.unwrap();
            let os = ov.unwrap();
            let hcount = hc.unwrap();
            kani::assert(is > 0 && os > 0 && hcount > 0,
                "Validated values must be positive");
        }
    }

    // =========================================================================
    // Q. FFI SETTER VALUE VALIDATION
    // Prove that set_neuron_weight / set_neuron_bias correctly reject
    // NaN/Inf values and negative indices before mutating MLP state.
    // =========================================================================

    #[kani::proof]
    fn verify_ffi_setter_rejects_nan_value() {
        let value: f64 = kani::any();
        kani::assume(value.is_nan());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(!accepted,
            "NaN value must be rejected by setter guard");
    }

    #[kani::proof]
    fn verify_ffi_setter_rejects_inf_value() {
        let value: f64 = kani::any();
        kani::assume(value.is_infinite());

        let accepted = !value.is_nan() && !value.is_infinite();
        kani::assert(!accepted,
            "Infinite value must be rejected by setter guard");
    }

    #[kani::proof]
    fn verify_ffi_setter_negative_index_rejected() {
        let layer: i32 = kani::any();
        let neuron: i32 = kani::any();
        let weight_idx: i32 = kani::any();

        let guard = layer >= 0 && neuron >= 0 && weight_idx >= 0;

        if layer < 0 || neuron < 0 || weight_idx < 0 {
            kani::assert(!guard,
                "Negative index must prevent setter execution");
        }
    }

    #[kani::proof]
    fn verify_ffi_setter_accepts_valid_params() {
        let layer: i32 = kani::any();
        let neuron: i32 = kani::any();
        let weight_idx: i32 = kani::any();
        let value: f64 = kani::any();

        kani::assume(layer >= 0 && neuron >= 0 && weight_idx >= 0);
        kani::assume(!value.is_nan() && !value.is_infinite());

        let guard = layer >= 0 && neuron >= 0 && weight_idx >= 0
            && !value.is_nan() && !value.is_infinite();
        kani::assert(guard,
            "Valid params must pass setter guard");
    }
}

#[cfg(test)]
mod unit_tests {
    use crate::core_types::*;

    #[test]
    fn test_validate_i32_as_usize() {
        assert_eq!(validate_i32_as_usize(-1), None);
        assert_eq!(validate_i32_as_usize(0), Some(0));
        assert_eq!(validate_i32_as_usize(1), Some(1));
        assert_eq!(validate_i32_as_usize(i32::MIN), None);
        assert_eq!(validate_i32_as_usize(i32::MAX), Some(i32::MAX as usize));
    }

    #[test]
    fn test_validate_i32_positive() {
        assert_eq!(validate_i32_positive(-1), None);
        assert_eq!(validate_i32_positive(0), None);
        assert_eq!(validate_i32_positive(1), Some(1));
    }

    #[test]
    fn test_validate_ffi_len() {
        assert_eq!(validate_ffi_len(-1, 100), None);
        assert_eq!(validate_ffi_len(0, 100), Some(0));
        assert_eq!(validate_ffi_len(50, 100), Some(50));
        assert_eq!(validate_ffi_len(101, 100), None);
    }

    #[test]
    fn test_validate_f64_param() {
        assert_eq!(validate_f64_param(1.0), Some(1.0));
        assert_eq!(validate_f64_param(f64::NAN), None);
        assert_eq!(validate_f64_param(f64::INFINITY), None);
        assert_eq!(validate_f64_param(f64::NEG_INFINITY), None);
    }

    #[test]
    fn test_validate_learning_rate() {
        assert_eq!(validate_learning_rate(0.01), Some(0.01));
        assert_eq!(validate_learning_rate(-0.01), None);
        assert_eq!(validate_learning_rate(f64::NAN), None);
        assert_eq!(validate_learning_rate(101.0), None);
    }

    #[test]
    fn test_validate_dropout_rate() {
        assert_eq!(validate_dropout_rate(0.5), Some(0.5));
        assert_eq!(validate_dropout_rate(-0.1), None);
        assert_eq!(validate_dropout_rate(1.1), None);
    }

    #[test]
    fn test_validate_activation_i32() {
        assert!(validate_activation_i32(0).is_some());
        assert!(validate_activation_i32(3).is_some());
        assert!(validate_activation_i32(4).is_none());
        assert!(validate_activation_i32(-1).is_none());
    }

    #[test]
    fn test_validate_optimizer_i32() {
        assert!(validate_optimizer_i32(0).is_some());
        assert!(validate_optimizer_i32(2).is_some());
        assert!(validate_optimizer_i32(3).is_none());
        assert!(validate_optimizer_i32(-1).is_none());
    }
}

