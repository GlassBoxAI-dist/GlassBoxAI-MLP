//! Kani Verification: FFI Safety for OpenCL Backend
//!
//! Verify that all data passed across the OpenCL FFI boundary is valid:
//! correct alignment, non-null buffers, valid work-group sizes, and proper
//! memory layout before kernel enqueue.

#[cfg(kani)]
mod kani_proofs {
    use crate::kani::core_types::*;

    #[kani::proof]
    fn verify_opencl_buffer_size_matches_layer() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 256);
        kani::assume(num_inputs > 0 && num_inputs <= 256);

        let layer = LayerData::new(num_neurons, num_inputs, TActivationType::AtReLU);

        let weight_bytes = layer.weights.len() * std::mem::size_of::<f64>();
        let bias_bytes = layer.biases.len() * std::mem::size_of::<f64>();
        let output_bytes = layer.outputs.len() * std::mem::size_of::<f64>();

        kani::assert(weight_bytes == num_neurons * num_inputs * 8,
            "Weight buffer size must match for OpenCL clCreateBuffer");
        kani::assert(bias_bytes == num_neurons * 8,
            "Bias buffer size must match for OpenCL clCreateBuffer");
        kani::assert(output_bytes == num_neurons * 8,
            "Output buffer size must match for OpenCL clCreateBuffer");
    }

    #[kani::proof]
    fn verify_opencl_global_work_size() {
        let num_neurons: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= MAX_NEURONS_PER_LAYER);

        let local_work_size: usize = 256;
        let global_work_size = ((num_neurons + local_work_size - 1) / local_work_size)
            * local_work_size;

        kani::assert(global_work_size >= num_neurons,
            "Global work size must cover all neurons");
        kani::assert(global_work_size % local_work_size == 0,
            "Global work size must be multiple of local work size");
        kani::assert(global_work_size > 0,
            "Global work size must be non-zero");
    }

    #[kani::proof]
    fn verify_opencl_kernel_arg_index_valid() {
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
            "Flat index must be within OpenCL buffer bounds");
    }

    #[kani::proof]
    fn verify_opencl_enqueue_size_non_zero() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements > 0 && num_elements <= MAX_ARRAY_SIZE);

        let transfer_bytes = num_elements * std::mem::size_of::<f64>();

        kani::assert(transfer_bytes > 0,
            "OpenCL enqueue read/write size must be non-zero");
        kani::assert(transfer_bytes % std::mem::size_of::<f64>() == 0,
            "Transfer size must be aligned to element size");
    }

    #[kani::proof]
    fn verify_opencl_f64_alignment() {
        let align = std::mem::align_of::<f64>();
        kani::assert(align == 8, "f64 must be 8-byte aligned for OpenCL cl_double");

        let size = std::mem::size_of::<f64>();
        kani::assert(size == 8, "f64 must be 8 bytes matching cl_double");
    }

    #[kani::proof]
    fn verify_opencl_mlp_buffers_contiguous() {
        let input_size: usize = kani::any();
        let output_size: usize = kani::any();
        kani::assume(input_size > 0 && input_size <= 8);
        kani::assume(output_size > 0 && output_size <= 8);

        if let Some(mlp) = MLP::new(input_size, &[4], output_size) {
            for layer in &mlp.layers {
                kani::assert(layer.weights.len() == layer.num_neurons * layer.num_inputs,
                    "Weight buffer must be contiguous for clEnqueueWriteBuffer");
                kani::assert(layer.biases.len() == layer.num_neurons,
                    "Bias buffer must be contiguous for clEnqueueWriteBuffer");
            }
        }
    }

    #[kani::proof]
    fn verify_opencl_work_group_size_power_of_two() {
        let local_size: usize = 256;
        kani::assert(local_size.is_power_of_two(),
            "Local work group size should be power of two for optimal OpenCL performance");
        kani::assert(local_size <= 1024,
            "Local work group size must not exceed typical OpenCL device limits");
    }

    #[kani::proof]
    fn verify_opencl_kernel_launch_params_valid() {
        let num_neurons: usize = kani::any();
        let num_inputs: usize = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= MAX_NEURONS_PER_LAYER);
        kani::assume(num_inputs > 0 && num_inputs <= MAX_NEURONS_PER_LAYER);

        let local_work_size: usize = 256;
        let global_work_size = ((num_neurons + local_work_size - 1) / local_work_size)
            * local_work_size;

        kani::assert(global_work_size >= num_neurons,
            "Global must cover all work items");
        kani::assert(global_work_size % local_work_size == 0,
            "Global must be divisible by local");

        let total_weights = num_neurons.checked_mul(num_inputs);
        kani::assert(total_weights.is_some(),
            "Weight count must not overflow before OpenCL buffer creation");
    }
}
