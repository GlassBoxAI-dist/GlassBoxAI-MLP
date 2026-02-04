.PHONY: build install test clean help

help:
	@echo "Facaded MLP Build Commands"
	@echo "=========================="
	@echo ""
	@echo "Rust builds:"
	@echo "  make build          - Build library (CUDA only)"
	@echo "  make build-opencl   - Build library (OpenCL only)"
	@echo "  make build-all      - Build library (CUDA + OpenCL)"
	@echo "  make build-cli      - Build CLI (CUDA + OpenCL)"
	@echo ""
	@echo "Python builds:"
	@echo "  make install        - Install Python package (release, all backends)"
	@echo "  make install-dev    - Install Python package (debug)"
	@echo "  make install-cpu    - Install Python package (CPU only)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run Python tests"
	@echo "  make run-xor        - Run XOR example"
	@echo "  make run-backends   - Run backend comparison example"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove build artifacts"

build:
	cargo build --release --features cuda

build-opencl:
	cargo build --release --features opencl

build-all:
	cargo build --release --features cuda,opencl

build-cli:
	cargo build --release --features cli,cuda,opencl

install:
	maturin develop --release --features python,cuda,opencl

install-dev:
	maturin develop --features python,cuda,opencl

install-cpu:
	maturin develop --release --features python

test: install
	pytest tests/ -v

run-xor: install
	python examples/xor_example.py

run-backends: install
	python examples/gpu_backend_example.py

clean:
	cargo clean
	rm -rf target/
	rm -rf *.so
	rm -rf python/facaded_mlp_cuda/*.so
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -f *.json *.onnx
