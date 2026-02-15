# Kani Verification Test Suite (CISA Hardening)

This directory contains formal verification harnesses for the Facaded MLP CUDA implementation, following **CISA "Secure by Design"** standards.

## Overview

The Kani Rust Verifier uses symbolic execution to mathematically prove that safety properties hold for **all possible inputs**, not just tested cases.

## Requirements

```bash
# Install Kani (requires Rust toolchain)
cargo install --locked kani-verifier
kani setup
```

## Running Verification

```bash
# Run all proofs
cd kani
cargo kani --tests

# Run a specific proof
cargo kani --harness verify_array_bounds_weight_access

# Run unit tests
cargo test
```

## Security Requirements Covered

### 1. Strict Bound Checks ✓
Proofs that all collection indexing is mathematically incapable of out-of-bounds access:
- `verify_array_bounds_layer_access`
- `verify_array_bounds_weight_access`
- `verify_array_bounds_bias_access`
- `verify_array_bounds_output_access`
- `verify_validate_bounds_generic`

### 2. Pointer Validity Proofs ✓
Verification that all memory accesses are valid and initialized:
- `verify_no_null_pointer_in_layer_creation`
- `verify_mlp_initialization_validity`

### 3. No-Panic Guarantee ✓
Proofs that functions cannot trigger panic across all inputs:
- `verify_activation_functions_no_panic`
- `verify_max_index_no_panic`
- `verify_mlp_construction_no_panic`
- `verify_parse_activation_no_panic`
- `verify_parse_optimizer_no_panic`

### 4. Integer Overflow Prevention ✓
Proofs that arithmetic operations are safe from overflow:
- `verify_safe_add_no_overflow`
- `verify_safe_sub_no_overflow`
- `verify_safe_mul_no_overflow`
- `verify_layer_size_calculation_no_overflow`

### 5. Division-by-Zero Exclusion ✓
Proofs that denominators are never zero:
- `verify_safe_div_no_zero`
- `verify_normalization_no_div_by_zero`
- `verify_softmax_denominator_non_zero`

### 6. Global State Consistency ✓
Proofs that mutations preserve invariants:
- `verify_mlp_invariants_after_mutation`
- `verify_layer_invariants_preserved`

### 7. Deadlock-Free Logic ✓
Verification of lock hierarchy patterns:
- `verify_no_reentrant_locking_pattern`

### 8. Input Sanitization Bounds ✓
Proofs that loops have formal upper bounds:
- `verify_bounded_loop_terminates`
- `verify_training_epoch_bounded`
- `verify_hidden_layer_count_bounded`

### 9. Result Coverage Audit ✓
Verification that all Result/Option types are handled:
- `verify_layer_access_result_handling`
- `verify_mlp_creation_result_handling`
- `verify_compute_loss_result_handling`

### 10. Memory Leak/Leakage Proofs ✓
Verification of memory ownership:
- `verify_layer_data_owned_vectors`
- `verify_allocation_with_limit_respects_budget`

### 11. Constant-Time Execution ✓
Proofs for timing-safe operations:
- `verify_sigmoid_constant_time_bounds`
- `verify_relu_constant_time_output`
- `verify_activation_selection_public_key`

### 12. State Machine Integrity ✓
Proofs preventing privilege escalation:
- `verify_privilege_escalation_blocked`
- `verify_unprivileged_cannot_escalate`

### 13. Enum Exhaustion ✓
Verification that match statements are exhaustive:
- `verify_activation_type_exhaustive`
- `verify_optimizer_type_exhaustive`
- `verify_command_type_exhaustive`

### 14. Floating-Point Sanity ✓
Proofs that NaN/Infinity states are handled:
- `verify_fp_sanity_check`
- `verify_clamp_fp_handles_special_values`
- `verify_sigmoid_never_nan_or_inf`
- `verify_relu_never_nan`
- `verify_compute_loss_nan_handling`

### 15. Resource Limit Compliance ✓
Proofs that allocations respect security budgets:
- `verify_memory_budget_enforcement`
- `verify_layer_allocation_within_budget`
- `verify_mlp_total_memory_bounded`

### 16. FFI Boundary Safety ✓
Proofs that all data crossing the C FFI boundary (julia.rs) is validated before use.
Covers the complete extern "C" surface consumed by C++, Go, C#, Julia, Zig, and Python wrappers.

#### A. Signed-to-Unsigned Conversion Safety
- `verify_i32_to_usize_rejects_negative`
- `verify_i32_positive_rejects_zero_and_negative`
- `verify_ffi_len_validates_range`
- `verify_ffi_len_i32_min_rejected`
- `verify_ffi_len_negative_one_rejected`

#### B. Output Buffer Overflow Prevention
- `verify_negative_capacity_prevents_buffer_write`
- `verify_zero_capacity_prevents_buffer_write`
- `verify_output_write_bounded_by_validated_capacity`

#### C. NaN/Infinity Parameter Rejection
- `verify_f64_param_rejects_nan`
- `verify_f64_param_rejects_infinity`
- `verify_f64_param_accepts_finite`
- `verify_learning_rate_validation`
- `verify_dropout_rate_validation`
- `verify_l2_lambda_validation`

#### D. Enum Variant Validation from Foreign Callers
- `verify_activation_i32_validation_exhaustive`
- `verify_activation_i32_negative_rejected`
- `verify_optimizer_i32_validation_exhaustive`
- `verify_optimizer_i32_negative_rejected`

#### E. MLP Creation Preconditions
- `verify_ffi_create_rejects_zero_input`
- `verify_ffi_create_rejects_zero_output`
- `verify_ffi_create_rejects_zero_hidden`
- `verify_ffi_create_rejects_excessive_hidden_layers`
- `verify_ffi_create_rejects_oversized_hidden`
- `verify_ffi_hidden_count_i32_negative_as_usize_huge`
- `verify_ffi_i32_min_as_usize_huge`

#### F. Train/Predict Length Validation
- `verify_ffi_train_input_len_validated`
- `verify_ffi_predict_capacity_validated`
- `verify_ffi_predict_output_bounded_by_capacity`

#### G. Layer/Neuron Index Validation
- `verify_ffi_layer_index_negative_rejected`
- `verify_ffi_layer_index_out_of_bounds_safe`
- `verify_ffi_neuron_index_negative_rejected`
- `verify_ffi_weight_index_negative_rejected`

#### H. Histogram Parameter Validation
- `verify_ffi_histogram_bins_negative_rejected`
- `verify_ffi_histogram_bins_zero_rejected`

#### I. Error String Safety
- `verify_ffi_error_nul_byte_sanitized`

#### J. No-Panic Guarantee for All FFI Validators
- `verify_validate_i32_as_usize_no_panic`
- `verify_validate_i32_positive_no_panic`
- `verify_validate_ffi_len_no_panic`
- `verify_validate_f64_param_no_panic`
- `verify_validate_f64_param_range_no_panic`
- `verify_validate_learning_rate_no_panic`
- `verify_validate_dropout_rate_no_panic`
- `verify_validate_l2_lambda_no_panic`
- `verify_validate_activation_i32_no_panic`
- `verify_validate_optimizer_i32_no_panic`

#### K. ABI Type Compatibility
- `verify_activation_type_repr_i32_abi`
- `verify_optimizer_type_repr_i32_abi`
- `verify_f64_abi_compatibility`
- `verify_i32_abi_compatibility`

#### L. Input Array NaN/Infinity Detection
- `verify_ffi_input_array_nan_detection`
- `verify_ffi_input_array_infinity_detection`

#### M. Resource Limits at Boundary
- `verify_ffi_allocation_respects_budget_at_boundary`
- `verify_ffi_to_vec_copy_bounded`

#### N. State Consistency After Parameter Mutation
- `verify_ffi_parameter_mutation_preserves_structure`

#### O. End-to-End FFI Pipeline Validation
- `verify_ffi_complete_train_validation_pipeline`
- `verify_ffi_complete_predict_validation_pipeline`
- `verify_ffi_complete_create_validation_pipeline`

#### P. FFI Setter Value Validation
- `verify_ffi_setter_rejects_nan_value`
- `verify_ffi_setter_rejects_inf_value`
- `verify_ffi_setter_negative_index_rejected`

## Architecture

```
kani/
├── Cargo.toml          # Standalone crate for verification
├── lib.rs              # Module root
├── core_types.rs       # Standalone types (no CUDA deps)
├── harnesses.rs        # All #[kani::proof] harnesses (categories 1-15)
├── ffi_boundary.rs     # FFI boundary safety harnesses (category 16)
└── README.md           # This file
```

## Design Decisions

1. **Standalone Crate**: The verification harnesses are in a separate crate to avoid CUDA dependencies during symbolic execution.

2. **Mirrored Types**: Core types from `mlp.rs` are mirrored in `core_types.rs` with Option-returning APIs for formal verification.

3. **Bounded Symbolic Inputs**: All proofs use `kani::assume` to constrain symbolic inputs to realistic ranges, ensuring tractable verification times.

4. **Safe Rust Focus**: Since the main crate uses safe Rust (except for CUDA FFI), many memory safety properties are guaranteed by the type system.

## Interpreting Results

- **VERIFICATION:- SUCCESSFUL**: The property holds for all possible inputs
- **VERIFICATION:- FAILED**: A counterexample was found; check the trace
- **TIMEOUT**: Increase unwind bounds or simplify the harness

## Integration with CI

```yaml
# Example GitHub Actions
kani-verification:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: model-checking/kani-github-action@v1
      with:
        working-directory: kani
```

## License

MIT License - Copyright (c) 2025 Matthew Abbott
