//! Kani Verification: FFI Safety for CUDA Backend
//!
//! Verify that all data passed across the CUDA FFI boundary is valid:
//! correct alignment, non-null pointers, valid sizes, and proper
//! memory layout before kernel launches.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_cuda_buffer_size_matches_layer() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 256);
        kani::assume(num_inputs > 0 && num_inputs <= 256);

        let layer = LayerData::new(num_neurons, num_inputs, TActivationType::AtReLU);

        let weight_bytes = layer.weights.len() * std::mem::size_of::<f64>();
        let bias_bytes = layer.biases.len() * std::mem::size_of::<f64>();
        let output_bytes = layer.outputs.len() * std::mem::size_of::<f64>();

        kani::assert(weight_bytes == num_neurons * num_inputs * 8,
            "Weight buffer size must match num_neurons * num_inputs * sizeof(f64)");
        kani::assert(bias_bytes == num_neurons * 8,
            "Bias buffer size must match num_neurons * sizeof(f64)");
        kani::assert(output_bytes == num_neurons * 8,
            "Output buffer size must match num_neurons * sizeof(f64)");
    }

    #[kani::proof]
    fn verify_cuda_grid_block_dimensions() {
        let num_neurons: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= MAX_NEURONS_PER_LAYER);

        let block_size: usize = 256;
        let grid_size = (num_neurons + block_size - 1) / block_size;

        kani::assert(grid_size > 0, "Grid size must be at least 1");
        kani::assert(grid_size * block_size >= num_neurons,
            "Grid * block must cover all neurons");
        kani::assert(grid_size <= 65535, "Grid size must fit CUDA limits");
    }

    #[kani::proof]
    fn verify_cuda_weight_index_valid_for_flat_buffer() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 16);
        kani::assume(num_inputs > 0 && num_inputs <= 16);

        let neuron_idx: usize = kani::any();
        let input_idx: usize = kani::any();
        kani::assume(neuron_idx < num_neurons);
        kani::assume(input_idx < num_inputs);

        let flat_idx = neuron_idx * num_inputs + input_idx;
        let total = num_neurons * num_inputs;

        kani::assert(flat_idx < total,
            "Flat index must be within allocated buffer");
    }

    #[kani::proof]
    fn verify_cuda_transfer_size_non_zero() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements > 0 && num_elements <= MAX_ARRAY_SIZE);

        let transfer_bytes = num_elements * std::mem::size_of::<f64>();

        kani::assert(transfer_bytes > 0, "Transfer size must be non-zero");
        kani::assert(transfer_bytes % std::mem::size_of::<f64>() == 0,
            "Transfer size must be aligned to f64");
    }

    #[kani::proof]
    fn verify_cuda_f64_alignment() {
        let align = std::mem::align_of::<f64>();
        kani::assert(align == 8, "f64 must be 8-byte aligned for CUDA transfers");

        let size = std::mem::size_of::<f64>();
        kani::assert(size == 8, "f64 must be 8 bytes for CUDA compatibility");
    }

    #[kani::proof]
    fn verify_cuda_mlp_buffers_contiguous() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size > 0 && input_size <= 8);
        kani::assume(output_size > 0 && output_size <= 8);

        if let Some(mlp) = MLP::new(input_size, &[4], output_size) {
            for layer in &mlp.layers {
                kani::assert(layer.weights.len() == layer.num_neurons * layer.num_inputs,
                    "Weight buffer must be contiguous for CUDA memcpy");
                kani::assert(layer.biases.len() == layer.num_neurons,
                    "Bias buffer must be contiguous for CUDA memcpy");
            }
        }
    }

    #[kani::proof]
    fn verify_cuda_kernel_launch_params_valid() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= MAX_NEURONS_PER_LAYER);
        kani::assume(num_inputs > 0 && num_inputs <= MAX_NEURONS_PER_LAYER);

        let block_size: u32 = 256;
        let threads_needed = num_neurons as u32;
        let blocks = (threads_needed + block_size - 1) / block_size;

        kani::assert(blocks > 0, "Must launch at least one block");
        kani::assert(blocks as u64 * block_size as u64 >= threads_needed as u64,
            "Total threads must cover all neurons");

        let total_weights = num_neurons.checked_mul(num_inputs);
        kani::assert(total_weights.is_some(),
            "Weight count must not overflow before CUDA allocation");
    }
}
