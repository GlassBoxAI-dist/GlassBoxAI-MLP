/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum TGPUBackend {
    CPU,
    CUDA,
    OpenCL,
}

impl TGPUBackend {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" => TGPUBackend::CUDA,
            "opencl" | "ocl" => TGPUBackend::OpenCL,
            "cpu" => TGPUBackend::CPU,
            _ => TGPUBackend::CPU,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            TGPUBackend::CPU => "cpu",
            TGPUBackend::CUDA => "cuda",
            TGPUBackend::OpenCL => "opencl",
        }
    }
}

pub fn detect_available_backends() -> Vec<TGPUBackend> {
    let mut backends = vec![TGPUBackend::CPU];
    
    #[cfg(feature = "cuda")]
    {
        if cudarc::driver::CudaDevice::new(0).is_ok() {
            backends.push(TGPUBackend::CUDA);
        }
    }
    
    #[cfg(feature = "opencl")]
    {
        if !ocl::Platform::list().is_empty() {
            backends.push(TGPUBackend::OpenCL);
        }
    }
    
    backends
}

pub fn select_best_backend() -> TGPUBackend {
    let available = detect_available_backends();
    
    // Prefer CUDA > OpenCL > CPU
    if available.contains(&TGPUBackend::CUDA) {
        TGPUBackend::CUDA
    } else if available.contains(&TGPUBackend::OpenCL) {
        TGPUBackend::OpenCL
    } else {
        TGPUBackend::CPU
    }
}
