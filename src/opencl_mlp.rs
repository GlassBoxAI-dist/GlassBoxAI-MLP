//! @file
//! @ingroup MLP_Internal_Logic
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

#[cfg(feature = "opencl")]
use ocl::{Platform, Device, Context, Queue, Program, Buffer, Kernel};
use crate::mlp::*;
use crate::opencl_kernels::OPENCL_KERNEL_SRC;

#[cfg(feature = "opencl")]
pub struct TOpenCLContext {
    context: Context,
    queue: Queue,
    program: Program,
}

#[cfg(feature = "opencl")]
impl TOpenCLContext {
    pub fn new() -> Result<Self, String> {
        let platform = Platform::default();
        let device = Device::first(platform).map_err(|e| format!("OpenCL device error: {}", e))?;
        
        let context = Context::builder()
            .platform(platform)
            .devices(device.clone())
            .build()
            .map_err(|e| format!("OpenCL context error: {}", e))?;
        
        let queue = Queue::new(&context, device, None)
            .map_err(|e| format!("OpenCL queue error: {}", e))?;
        
        let program = Program::builder()
            .devices(device)
            .src(OPENCL_KERNEL_SRC)
            .build(&context)
            .map_err(|e| format!("OpenCL program build error: {}", e))?;
        
        Ok(TOpenCLContext {
            context,
            queue,
            program,
        })
    }
    
    pub fn feed_forward_layer(
        &self,
        input: &Darray,
        weights: &Darray,
        biases: &Darray,
        output_size: usize,
        activation_type: i32,
    ) -> Result<Darray, String> {
        let input_size = input.len();
        let mut output = vec![0.0; output_size];
        
        let input_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(input_size)
            .copy_host_slice(input)
            .build()
            .map_err(|e| format!("Buffer create error: {}", e))?;
        
        let weights_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(weights.len())
            .copy_host_slice(weights)
            .build()
            .map_err(|e| format!("Buffer create error: {}", e))?;
        
        let biases_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(biases.len())
            .copy_host_slice(biases)
            .build()
            .map_err(|e| format!("Buffer create error: {}", e))?;
        
        let output_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(output_size)
            .build()
            .map_err(|e| format!("Buffer create error: {}", e))?;
        
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("feedForwardLayer")
            .queue(self.queue.clone())
            .global_work_size(output_size)
            .arg(&input_buffer)
            .arg(&weights_buffer)
            .arg(&biases_buffer)
            .arg(&output_buffer)
            .arg(input_size as i32)
            .arg(output_size as i32)
            .arg(activation_type)
            .build()
            .map_err(|e| format!("Kernel build error: {}", e))?;
        
        unsafe {
            kernel.enq().map_err(|e| format!("Kernel execution error: {}", e))?;
        }
        
        output_buffer.read(&mut output).enq()
            .map_err(|e| format!("Buffer read error: {}", e))?;
        
        Ok(output)
    }
    
    pub fn softmax(&self, input: &Darray) -> Result<Darray, String> {
        let size = input.len();
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = input.iter().map(|&x| (x - max_val).exp()).sum();
        
        let input_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(size)
            .copy_host_slice(input)
            .build()
            .map_err(|e| format!("Buffer create error: {}", e))?;
        
        let output_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::flags::MEM_WRITE_ONLY)
            .len(size)
            .build()
            .map_err(|e| format!("Buffer create error: {}", e))?;
        
        let kernel = Kernel::builder()
            .program(&self.program)
            .name("softmaxKernel")
            .queue(self.queue.clone())
            .global_work_size(size)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(size as i32)
            .arg(max_val)
            .arg(sum_exp)
            .build()
            .map_err(|e| format!("Kernel build error: {}", e))?;
        
        unsafe {
            kernel.enq().map_err(|e| format!("Kernel execution error: {}", e))?;
        }
        
        let mut output = vec![0.0; size];
        output_buffer.read(&mut output).enq()
            .map_err(|e| format!("Buffer read error: {}", e))?;
        
        Ok(output)
    }
}

#[cfg(not(feature = "opencl"))]
pub struct TOpenCLContext;

#[cfg(not(feature = "opencl"))]
impl TOpenCLContext {
    pub fn new() -> Result<Self, String> {
        Err("OpenCL support not compiled".to_string())
    }
}

