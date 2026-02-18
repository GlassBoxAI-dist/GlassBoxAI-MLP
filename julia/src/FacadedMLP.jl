## @file
## @ingroup MLP_Wrappers
"""
    FacadedMLP

GPU-accelerated Multi-Layer Perceptron implementation with CUDA/OpenCL/CPU backends.

# Example
```julia
using FacadedMLP

# Create a network: 2 inputs, 8 hidden neurons, 1 output
mlp = MLP(2, [8], 1)

# Train on XOR
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]

losses = fit!(mlp, X, y; epochs=1000, verbose=true)

# Predict
output = predict(mlp, [1.0, 0.0])
```
"""
module FacadedMLP

export MLP, ActivationType, OptimizerType
export predict, train!, fit!, predict_batch
export save, load, feature_importance
export available_backends, set_backend!
export get_neuron_weights, get_neuron_bias, set_neuron_weight!, set_neuron_bias!
export get_layer_outputs
export compute_loss, get_layer_errors, get_layer_size, get_layer_activation, get_layer_info
export get_neuron_view, get_weight_m, get_weight_v, get_bias_m, get_bias_v
export get_activation_histogram, get_gradient_histogram
export export_onnx, import_onnx

# Activation types
@enum ActivationType begin
    Sigmoid = 0
    Tanh = 1
    ReLU = 2
    Softmax = 3
end

# Optimizer types
@enum OptimizerType begin
    SGD = 0
    Adam = 1
    RMSProp = 2
end

# Find the library
function find_library()
    # Check common locations
    candidates = [
        # Development build
        joinpath(@__DIR__, "..", "..", "target", "release", "libfacaded_mlp_cuda.so"),
        joinpath(@__DIR__, "..", "..", "target", "release", "libfacaded_mlp_cuda.dylib"),
        joinpath(@__DIR__, "..", "..", "target", "release", "facaded_mlp_cuda.dll"),
        joinpath(@__DIR__, "..", "..", "target", "debug", "libfacaded_mlp_cuda.so"),
        joinpath(@__DIR__, "..", "..", "target", "debug", "libfacaded_mlp_cuda.dylib"),
        joinpath(@__DIR__, "..", "..", "target", "debug", "facaded_mlp_cuda.dll"),
        # Installed location
        joinpath(@__DIR__, "..", "lib", "libfacaded_mlp_cuda.so"),
        joinpath(@__DIR__, "..", "lib", "libfacaded_mlp_cuda.dylib"),
        joinpath(@__DIR__, "..", "lib", "facaded_mlp_cuda.dll"),
    ]
    
    for path in candidates
        if isfile(path)
            return path
        end
    end
    
    # Try system path
    return "libfacaded_mlp_cuda"
end

const LIBMLP = Ref{String}()

function __init__()
    LIBMLP[] = find_library()
end

# C FFI wrappers

function c_create(input_size::Int32, hidden_sizes::Vector{Int32}, output_size::Int32,
                  hidden_activation::Int32, output_activation::Int32, backend::String)
    ptr = ccall((:mlp_create, LIBMLP[]), Ptr{Cvoid},
        (Int32, Ptr{Int32}, Int32, Int32, Int32, Int32, Cstring),
        input_size, hidden_sizes, Int32(length(hidden_sizes)), output_size,
        hidden_activation, output_activation, backend)
    if ptr == C_NULL
        error_msg = c_get_last_error()
        error("Failed to create MLP: $error_msg")
    end
    ptr
end

function c_destroy(ptr::Ptr{Cvoid})
    ccall((:mlp_destroy, LIBMLP[]), Cvoid, (Ptr{Cvoid},), ptr)
end

function c_train(ptr::Ptr{Cvoid}, input::Vector{Float64}, target::Vector{Float64})
    status = ccall((:mlp_train, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Int32, Ptr{Float64}, Int32),
        ptr, input, Int32(length(input)), target, Int32(length(target)))
    if status != 0
        error_msg = c_get_last_error()
        error("Training failed: $error_msg")
    end
end

function c_predict(ptr::Ptr{Cvoid}, input::Vector{Float64}, output_size::Int)
    output = Vector{Float64}(undef, output_size)
    len = ccall((:mlp_predict, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Int32, Ptr{Float64}, Int32),
        ptr, input, Int32(length(input)), output, Int32(output_size))
    if len < 0
        error_msg = c_get_last_error()
        error("Prediction failed: $error_msg")
    end
    resize!(output, len)
    output
end

function c_compute_loss(ptr::Ptr{Cvoid}, output::Vector{Float64}, target::Vector{Float64})
    ccall((:mlp_compute_loss, LIBMLP[]), Float64,
        (Ptr{Cvoid}, Ptr{Float64}, Int32, Ptr{Float64}, Int32),
        ptr, output, Int32(length(output)), target, Int32(length(target)))
end

function c_save(ptr::Ptr{Cvoid}, filename::String)
    status = ccall((:mlp_save, LIBMLP[]), Int32, (Ptr{Cvoid}, Cstring), ptr, filename)
    if status != 0
        error_msg = c_get_last_error()
        error("Save failed: $error_msg")
    end
end

function c_load(filename::String)
    ptr = ccall((:mlp_load, LIBMLP[]), Ptr{Cvoid}, (Cstring,), filename)
    if ptr == C_NULL
        error_msg = c_get_last_error()
        error("Load failed: $error_msg")
    end
    ptr
end

function c_get_last_error()
    ptr = ccall((:mlp_get_last_error, LIBMLP[]), Ptr{Cchar}, ())
    ptr == C_NULL ? "Unknown error" : unsafe_string(ptr)
end

function c_get_learning_rate(ptr::Ptr{Cvoid})
    ccall((:mlp_get_learning_rate, LIBMLP[]), Float64, (Ptr{Cvoid},), ptr)
end

function c_set_learning_rate(ptr::Ptr{Cvoid}, value::Float64)
    ccall((:mlp_set_learning_rate, LIBMLP[]), Cvoid, (Ptr{Cvoid}, Float64), ptr, value)
end

function c_get_optimizer(ptr::Ptr{Cvoid})
    ccall((:mlp_get_optimizer, LIBMLP[]), Int32, (Ptr{Cvoid},), ptr)
end

function c_set_optimizer(ptr::Ptr{Cvoid}, value::Int32)
    ccall((:mlp_set_optimizer, LIBMLP[]), Cvoid, (Ptr{Cvoid}, Int32), ptr, value)
end

function c_get_dropout_rate(ptr::Ptr{Cvoid})
    ccall((:mlp_get_dropout_rate, LIBMLP[]), Float64, (Ptr{Cvoid},), ptr)
end

function c_set_dropout_rate(ptr::Ptr{Cvoid}, value::Float64)
    ccall((:mlp_set_dropout_rate, LIBMLP[]), Cvoid, (Ptr{Cvoid}, Float64), ptr, value)
end

function c_get_l2_lambda(ptr::Ptr{Cvoid})
    ccall((:mlp_get_l2_lambda, LIBMLP[]), Float64, (Ptr{Cvoid},), ptr)
end

function c_set_l2_lambda(ptr::Ptr{Cvoid}, value::Float64)
    ccall((:mlp_set_l2_lambda, LIBMLP[]), Cvoid, (Ptr{Cvoid}, Float64), ptr, value)
end

function c_get_batch_norm(ptr::Ptr{Cvoid})
    ccall((:mlp_get_batch_norm, LIBMLP[]), Int32, (Ptr{Cvoid},), ptr) != 0
end

function c_set_batch_norm(ptr::Ptr{Cvoid}, value::Bool)
    ccall((:mlp_set_batch_norm, LIBMLP[]), Cvoid, (Ptr{Cvoid}, Int32), ptr, value ? 1 : 0)
end

function c_get_input_size(ptr::Ptr{Cvoid})
    Int(ccall((:mlp_get_input_size, LIBMLP[]), Int32, (Ptr{Cvoid},), ptr))
end

function c_get_output_size(ptr::Ptr{Cvoid})
    Int(ccall((:mlp_get_output_size, LIBMLP[]), Int32, (Ptr{Cvoid},), ptr))
end

function c_get_num_layers(ptr::Ptr{Cvoid})
    Int(ccall((:mlp_get_num_layers, LIBMLP[]), Int32, (Ptr{Cvoid},), ptr))
end

function c_get_hidden_sizes(ptr::Ptr{Cvoid}, max_layers::Int)
    sizes = Vector{Int32}(undef, max_layers)
    len = ccall((:mlp_get_hidden_sizes, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Ptr{Int32}, Int32), ptr, sizes, Int32(max_layers))
    resize!(sizes, len)
    Int.(sizes)
end

function c_get_backend(ptr::Ptr{Cvoid})
    ptr_str = ccall((:mlp_get_backend, LIBMLP[]), Ptr{Cchar}, (Ptr{Cvoid},), ptr)
    ptr_str == C_NULL ? "unknown" : unsafe_string(ptr_str)
end

function c_set_backend(ptr::Ptr{Cvoid}, backend::String)
    status = ccall((:mlp_set_backend, LIBMLP[]), Int32, (Ptr{Cvoid}, Cstring), ptr, backend)
    if status != 0
        error_msg = c_get_last_error()
        error("Set backend failed: $error_msg")
    end
end

function c_available_backends()
    ptr = ccall((:mlp_available_backends, LIBMLP[]), Ptr{Cchar}, ())
    if ptr == C_NULL
        return String[]
    end
    result = unsafe_string(ptr)
    ccall((:mlp_free_string, LIBMLP[]), Cvoid, (Ptr{Cchar},), ptr)
    split(result, ",")
end

function c_get_neuron_weights(ptr::Ptr{Cvoid}, layer::Int, neuron::Int, max_weights::Int)
    weights = Vector{Float64}(undef, max_weights)
    len = ccall((:mlp_get_neuron_weights, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Int32, Int32, Ptr{Float64}, Int32),
        ptr, Int32(layer), Int32(neuron), weights, Int32(max_weights))
    resize!(weights, len)
    weights
end

function c_get_neuron_bias(ptr::Ptr{Cvoid}, layer::Int, neuron::Int)
    ccall((:mlp_get_neuron_bias, LIBMLP[]), Float64,
        (Ptr{Cvoid}, Int32, Int32), ptr, Int32(layer), Int32(neuron))
end

function c_set_neuron_weight(ptr::Ptr{Cvoid}, layer::Int, neuron::Int, weight_idx::Int, value::Float64)
    ccall((:mlp_set_neuron_weight, LIBMLP[]), Cvoid,
        (Ptr{Cvoid}, Int32, Int32, Int32, Float64),
        ptr, Int32(layer), Int32(neuron), Int32(weight_idx), value)
end

function c_set_neuron_bias(ptr::Ptr{Cvoid}, layer::Int, neuron::Int, value::Float64)
    ccall((:mlp_set_neuron_bias, LIBMLP[]), Cvoid,
        (Ptr{Cvoid}, Int32, Int32, Float64), ptr, Int32(layer), Int32(neuron), value)
end

function c_get_layer_outputs(ptr::Ptr{Cvoid}, layer::Int, max_outputs::Int)
    outputs = Vector{Float64}(undef, max_outputs)
    len = ccall((:mlp_get_layer_outputs, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Int32, Ptr{Float64}, Int32),
        ptr, Int32(layer), outputs, Int32(max_outputs))
    resize!(outputs, len)
    outputs
end

function c_feature_importance(ptr::Ptr{Cvoid}, input_size::Int)
    indices = Vector{Int32}(undef, input_size)
    scores = Vector{Float64}(undef, input_size)
    len = ccall((:mlp_feature_importance, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Ptr{Int32}, Ptr{Float64}, Int32),
        ptr, indices, scores, Int32(input_size))
    resize!(indices, len)
    resize!(scores, len)
    [(Int(idx), score) for (idx, score) in zip(indices, scores)]
end

function c_get_layer_errors(ptr::Ptr{Cvoid}, layer::Int, max_size::Int)
    errs = Vector{Float64}(undef, max_size)
    len = ccall((:mlp_get_layer_errors, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Int32, Ptr{Float64}, Int32),
        ptr, Int32(layer), errs, Int32(max_size))
    resize!(errs, max(0, len))
    errs
end

function c_get_layer_size(ptr::Ptr{Cvoid}, layer::Int)
    Int(ccall((:mlp_get_layer_size, LIBMLP[]), Int32, (Ptr{Cvoid}, Int32), ptr, Int32(layer)))
end

function c_get_layer_activation(ptr::Ptr{Cvoid}, layer::Int)
    ccall((:mlp_get_layer_activation, LIBMLP[]), Int32, (Ptr{Cvoid}, Int32), ptr, Int32(layer))
end

function c_get_weight_m(ptr::Ptr{Cvoid}, layer::Int, neuron::Int, weight_idx::Int)
    ccall((:mlp_get_weight_m, LIBMLP[]), Float64,
        (Ptr{Cvoid}, Int32, Int32, Int32), ptr, Int32(layer), Int32(neuron), Int32(weight_idx))
end

function c_get_weight_v(ptr::Ptr{Cvoid}, layer::Int, neuron::Int, weight_idx::Int)
    ccall((:mlp_get_weight_v, LIBMLP[]), Float64,
        (Ptr{Cvoid}, Int32, Int32, Int32), ptr, Int32(layer), Int32(neuron), Int32(weight_idx))
end

function c_get_bias_m(ptr::Ptr{Cvoid}, layer::Int, neuron::Int)
    ccall((:mlp_get_bias_m, LIBMLP[]), Float64,
        (Ptr{Cvoid}, Int32, Int32), ptr, Int32(layer), Int32(neuron))
end

function c_get_bias_v(ptr::Ptr{Cvoid}, layer::Int, neuron::Int)
    ccall((:mlp_get_bias_v, LIBMLP[]), Float64,
        (Ptr{Cvoid}, Int32, Int32), ptr, Int32(layer), Int32(neuron))
end

function c_get_activation_histogram(ptr::Ptr{Cvoid}, layer::Int, bins::Int)
    hist = Vector{Int32}(undef, bins)
    len = ccall((:mlp_get_activation_histogram, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Int32, Int32, Ptr{Int32}, Int32),
        ptr, Int32(layer), Int32(bins), hist, Int32(bins))
    resize!(hist, max(0, len))
    Int.(hist)
end

function c_get_gradient_histogram(ptr::Ptr{Cvoid}, layer::Int, bins::Int)
    hist = Vector{Int32}(undef, bins)
    len = ccall((:mlp_get_gradient_histogram, LIBMLP[]), Int32,
        (Ptr{Cvoid}, Int32, Int32, Ptr{Int32}, Int32),
        ptr, Int32(layer), Int32(bins), hist, Int32(bins))
    resize!(hist, max(0, len))
    Int.(hist)
end

function c_get_timestep(ptr::Ptr{Cvoid})
    Int(ccall((:mlp_get_timestep, LIBMLP[]), Int32, (Ptr{Cvoid},), ptr))
end

function c_export_onnx(ptr::Ptr{Cvoid}, filename::String)
    status = ccall((:mlp_export_onnx, LIBMLP[]), Int32, (Ptr{Cvoid}, Cstring), ptr, filename)
    if status != 0
        error_msg = c_get_last_error()
        error("ONNX export failed: $error_msg")
    end
end

function c_import_onnx(filename::String, backend::String)
    ptr = ccall((:mlp_import_onnx, LIBMLP[]), Ptr{Cvoid}, (Cstring, Cstring), filename, backend)
    if ptr == C_NULL
        error_msg = c_get_last_error()
        error("ONNX import failed: $error_msg")
    end
    ptr
end

# MLP struct

"""
    MLP

GPU-accelerated Multi-Layer Perceptron.

# Fields
- `input_size::Int`: Number of input neurons
- `output_size::Int`: Number of output neurons
- `hidden_sizes::Vector{Int}`: Hidden layer sizes
"""
mutable struct MLP
    ptr::Ptr{Cvoid}
    input_size::Int
    output_size::Int
    hidden_sizes::Vector{Int}
    
    function MLP(ptr::Ptr{Cvoid})
        mlp = new(ptr)
        mlp.input_size = c_get_input_size(ptr)
        mlp.output_size = c_get_output_size(ptr)
        mlp.hidden_sizes = c_get_hidden_sizes(ptr, 100)
        finalizer(mlp) do m
            c_destroy(m.ptr)
        end
        mlp
    end
end

"""
    MLP(input_size, hidden_sizes, output_size; kwargs...)

Create a new MLP model.

# Arguments
- `input_size::Int`: Number of input neurons
- `hidden_sizes::Vector{Int}`: List of hidden layer sizes
- `output_size::Int`: Number of output neurons

# Keyword Arguments
- `hidden_activation::ActivationType = Sigmoid`: Activation for hidden layers
- `output_activation::ActivationType = Sigmoid`: Activation for output layer
- `backend::String = "auto"`: GPU backend ("auto", "cpu", "cuda", "opencl")
- `learning_rate::Float64 = 0.01`: Learning rate
- `optimizer::OptimizerType = Adam`: Optimizer type
- `dropout_rate::Float64 = 0.0`: Dropout rate
- `l2_lambda::Float64 = 0.0`: L2 regularization
- `batch_norm::Bool = false`: Use batch normalization

# Example
```julia
mlp = MLP(2, [8, 8], 1; hidden_activation=ReLU, backend="cuda")
```
"""
function MLP(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int;
             hidden_activation::ActivationType = Sigmoid,
             output_activation::ActivationType = Sigmoid,
             backend::String = "auto",
             learning_rate::Float64 = 0.01,
             optimizer::OptimizerType = Adam,
             dropout_rate::Float64 = 0.0,
             l2_lambda::Float64 = 0.0,
             batch_norm::Bool = false)
    
    ptr = c_create(
        Int32(input_size),
        Int32.(hidden_sizes),
        Int32(output_size),
        Int32(hidden_activation),
        Int32(output_activation),
        backend
    )
    
    mlp = MLP(ptr)
    
    # Set additional parameters
    c_set_learning_rate(ptr, learning_rate)
    c_set_optimizer(ptr, Int32(optimizer))
    c_set_dropout_rate(ptr, dropout_rate)
    c_set_l2_lambda(ptr, l2_lambda)
    c_set_batch_norm(ptr, batch_norm)
    
    mlp
end

# Convenience constructor with single hidden layer
function MLP(input_size::Int, hidden_size::Int, output_size::Int; kwargs...)
    MLP(input_size, [hidden_size], output_size; kwargs...)
end

# Properties
Base.propertynames(::MLP) = (:learning_rate, :optimizer, :dropout_rate, :l2_lambda,
                              :batch_norm, :backend, :num_layers, :input_size,
                              :output_size, :hidden_sizes, :timestep)

function Base.getproperty(mlp::MLP, name::Symbol)
    if name === :ptr || name === :input_size || name === :output_size || name === :hidden_sizes
        return getfield(mlp, name)
    elseif name === :learning_rate
        return c_get_learning_rate(mlp.ptr)
    elseif name === :optimizer
        return OptimizerType(c_get_optimizer(mlp.ptr))
    elseif name === :dropout_rate
        return c_get_dropout_rate(mlp.ptr)
    elseif name === :l2_lambda
        return c_get_l2_lambda(mlp.ptr)
    elseif name === :batch_norm
        return c_get_batch_norm(mlp.ptr)
    elseif name === :backend
        return c_get_backend(mlp.ptr)
    elseif name === :num_layers
        return c_get_num_layers(mlp.ptr)
    elseif name === :timestep
        return c_get_timestep(mlp.ptr)
    else
        error("Unknown property: $name")
    end
end

function Base.setproperty!(mlp::MLP, name::Symbol, value)
    if name === :ptr || name === :input_size || name === :output_size || name === :hidden_sizes
        return setfield!(mlp, name, value)
    elseif name === :learning_rate
        c_set_learning_rate(mlp.ptr, Float64(value))
    elseif name === :optimizer
        c_set_optimizer(mlp.ptr, Int32(value))
    elseif name === :dropout_rate
        c_set_dropout_rate(mlp.ptr, Float64(value))
    elseif name === :l2_lambda
        c_set_l2_lambda(mlp.ptr, Float64(value))
    elseif name === :batch_norm
        c_set_batch_norm(mlp.ptr, Bool(value))
    else
        error("Cannot set property: $name")
    end
end

function Base.show(io::IO, mlp::MLP)
    print(io, "MLP(input=$(mlp.input_size), hidden=$(mlp.hidden_sizes), ",
          "output=$(mlp.output_size), lr=$(round(mlp.learning_rate, digits=4)), ",
          "optimizer=$(mlp.optimizer), backend=$(mlp.backend))")
end

# Core functions

"""
    predict(mlp, input) -> Vector{Float64}

Make a prediction with the model.

# Example
```julia
output = predict(mlp, [1.0, 0.0])
```
"""
function predict(mlp::MLP, input::AbstractVector{<:Real})
    c_predict(mlp.ptr, Float64.(input), mlp.output_size)
end

"""
    train!(mlp, input, target)

Train on a single sample.

# Example
```julia
train!(mlp, [1.0, 0.0], [1.0])
```
"""
function train!(mlp::MLP, input::AbstractVector{<:Real}, target::AbstractVector{<:Real})
    c_train(mlp.ptr, Float64.(input), Float64.(target))
end

"""
    fit!(mlp, inputs, targets; epochs=100, verbose=false) -> Vector{Float64}

Train on a dataset for multiple epochs.

# Arguments
- `inputs`: Vector of input vectors
- `targets`: Vector of target vectors
- `epochs::Int = 100`: Number of training epochs
- `verbose::Bool = false`: Print progress

# Returns
Loss per epoch.

# Example
```julia
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]
losses = fit!(mlp, X, y; epochs=1000, verbose=true)
```
"""
function fit!(mlp::MLP, inputs::AbstractVector, targets::AbstractVector;
              epochs::Int = 100, verbose::Bool = false)
    
    length(inputs) == length(targets) || error("inputs and targets must have same length")
    
    losses = Float64[]
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        
        for (input, target) in zip(inputs, targets)
            train!(mlp, input, target)
            output = predict(mlp, input)
            epoch_loss += c_compute_loss(mlp.ptr, output, Float64.(target))
        end
        
        epoch_loss /= length(inputs)
        push!(losses, epoch_loss)
        
        if verbose && (epoch % 100 == 0 || epoch == epochs)
            println("Epoch $epoch/$epochs - Loss: $(round(epoch_loss, digits=6))")
        end
    end
    
    losses
end

"""
    predict_batch(mlp, inputs) -> Vector{Vector{Float64}}

Predict on multiple samples.

# Example
```julia
outputs = predict_batch(mlp, [[0.0, 0.0], [1.0, 1.0]])
```
"""
function predict_batch(mlp::MLP, inputs::AbstractVector)
    [predict(mlp, input) for input in inputs]
end

"""
    save(mlp, filename)

Save model to file.

# Example
```julia
save(mlp, "model.json")
```
"""
function save(mlp::MLP, filename::AbstractString)
    c_save(mlp.ptr, String(filename))
end

"""
    load(filename) -> MLP

Load model from file.

# Example
```julia
mlp = load("model.json")
```
"""
function load(filename::AbstractString)
    ptr = c_load(String(filename))
    MLP(ptr)
end

"""
    feature_importance(mlp) -> Vector{Tuple{Int, Float64}}

Compute feature importance based on weights.

Returns vector of (feature_index, importance_score) tuples, sorted by importance.

# Example
```julia
importance = feature_importance(mlp)
for (idx, score) in importance
    println("Feature \$idx: \$score")
end
```
"""
function feature_importance(mlp::MLP)
    c_feature_importance(mlp.ptr, mlp.input_size)
end

"""
    available_backends() -> Vector{String}

Get list of available GPU backends.

# Example
```julia
backends = available_backends()  # e.g., ["cpu", "cuda"]
```
"""
function available_backends()
    String.(c_available_backends())
end

"""
    set_backend!(mlp, backend)

Change the GPU backend.

# Arguments
- `backend::String`: "cpu", "cuda", or "opencl"

# Example
```julia
set_backend!(mlp, "cuda")
```
"""
function set_backend!(mlp::MLP, backend::AbstractString)
    c_set_backend(mlp.ptr, String(backend))
end

"""
    get_neuron_weights(mlp, layer, neuron) -> Vector{Float64}

Get weights for a specific neuron.

# Arguments
- `layer::Int`: Layer index (1-indexed)
- `neuron::Int`: Neuron index (1-indexed)
"""
function get_neuron_weights(mlp::MLP, layer::Int, neuron::Int)
    prev_size = layer == 1 ? mlp.input_size : mlp.hidden_sizes[layer - 1]
    c_get_neuron_weights(mlp.ptr, layer, neuron, prev_size)
end

"""
    get_neuron_bias(mlp, layer, neuron) -> Float64

Get bias for a specific neuron.
"""
function get_neuron_bias(mlp::MLP, layer::Int, neuron::Int)
    c_get_neuron_bias(mlp.ptr, layer, neuron)
end

"""
    set_neuron_weight!(mlp, layer, neuron, weight_idx, value)

Set a specific weight.
"""
function set_neuron_weight!(mlp::MLP, layer::Int, neuron::Int, weight_idx::Int, value::Real)
    c_set_neuron_weight(mlp.ptr, layer, neuron, weight_idx, Float64(value))
end

"""
    set_neuron_bias!(mlp, layer, neuron, value)

Set a neuron's bias.
"""
function set_neuron_bias!(mlp::MLP, layer::Int, neuron::Int, value::Real)
    c_set_neuron_bias(mlp.ptr, layer, neuron, Float64(value))
end

"""
    get_layer_outputs(mlp, layer) -> Vector{Float64}

Get output values for all neurons in a layer.
"""
function get_layer_outputs(mlp::MLP, layer::Int)
    layer_size = if layer == 0
        mlp.input_size
    elseif layer <= length(mlp.hidden_sizes)
        mlp.hidden_sizes[layer]
    else
        mlp.output_size
    end
    c_get_layer_outputs(mlp.ptr, layer, layer_size)
end

"""
    compute_loss(mlp, output, target) -> Float64

Compute the loss between model output and target vectors.
"""
function compute_loss(mlp::MLP, output::AbstractVector{<:Real}, target::AbstractVector{<:Real})
    c_compute_loss(mlp.ptr, Float64.(output), Float64.(target))
end

"""
    get_layer_size(mlp, layer) -> Int

Return the number of neurons in a layer.
"""
function get_layer_size(mlp::MLP, layer::Int)
    c_get_layer_size(mlp.ptr, layer)
end

"""
    get_layer_activation(mlp, layer) -> ActivationType

Return the activation function used by a layer.
"""
function get_layer_activation(mlp::MLP, layer::Int)
    ActivationType(c_get_layer_activation(mlp.ptr, layer))
end

"""
    get_layer_errors(mlp, layer) -> Vector{Float64}

Return the error/gradient values for all neurons in a layer after the last training step.
"""
function get_layer_errors(mlp::MLP, layer::Int)
    size = c_get_layer_size(mlp.ptr, layer)
    size <= 0 && return Float64[]
    c_get_layer_errors(mlp.ptr, layer, size)
end

"""
    get_layer_info(mlp, layer) -> NamedTuple

Return combined metadata about a layer: `(index, size, activation)`.
"""
function get_layer_info(mlp::MLP, layer::Int)
    (index = layer,
     size = c_get_layer_size(mlp.ptr, layer),
     activation = ActivationType(c_get_layer_activation(mlp.ptr, layer)))
end

"""
    get_neuron_view(mlp, layer, neuron) -> NamedTuple

Return a detailed per-neuron view: `(layer, neuron, weights, bias, output, error)`.
"""
function get_neuron_view(mlp::MLP, layer::Int, neuron::Int)
    weights = get_neuron_weights(mlp, layer, neuron)
    bias    = c_get_neuron_bias(mlp.ptr, layer, neuron)
    outputs = get_layer_outputs(mlp, layer)
    errors  = get_layer_errors(mlp, layer)
    out_val = neuron <= length(outputs) ? outputs[neuron] : 0.0
    err_val = neuron <= length(errors)  ? errors[neuron]  : 0.0
    (layer = layer, neuron = neuron, weights = weights,
     bias = bias, output = out_val, error = err_val)
end

"""
    get_weight_m(mlp, layer, neuron, weight_idx) -> Float64

Return the Adam first moment (M) for a specific weight.
"""
function get_weight_m(mlp::MLP, layer::Int, neuron::Int, weight_idx::Int)
    c_get_weight_m(mlp.ptr, layer, neuron, weight_idx)
end

"""
    get_weight_v(mlp, layer, neuron, weight_idx) -> Float64

Return the Adam second moment (V) for a specific weight.
"""
function get_weight_v(mlp::MLP, layer::Int, neuron::Int, weight_idx::Int)
    c_get_weight_v(mlp.ptr, layer, neuron, weight_idx)
end

"""
    get_bias_m(mlp, layer, neuron) -> Float64

Return the Adam first moment (M) for a specific bias.
"""
function get_bias_m(mlp::MLP, layer::Int, neuron::Int)
    c_get_bias_m(mlp.ptr, layer, neuron)
end

"""
    get_bias_v(mlp, layer, neuron) -> Float64

Return the Adam second moment (V) for a specific bias.
"""
function get_bias_v(mlp::MLP, layer::Int, neuron::Int)
    c_get_bias_v(mlp.ptr, layer, neuron)
end

"""
    get_activation_histogram(mlp, layer, bins) -> Vector{Int}

Return a histogram of activation values across all neurons in a layer.
"""
function get_activation_histogram(mlp::MLP, layer::Int, bins::Int)
    c_get_activation_histogram(mlp.ptr, layer, bins)
end

"""
    get_gradient_histogram(mlp, layer, bins) -> Vector{Int}

Return a histogram of gradient values across all neurons in a layer.
"""
function get_gradient_histogram(mlp::MLP, layer::Int, bins::Int)
    c_get_gradient_histogram(mlp.ptr, layer, bins)
end

"""
    export_onnx(mlp, filename)

Export the model to an ONNX file.
"""
function export_onnx(mlp::MLP, filename::AbstractString)
    c_export_onnx(mlp.ptr, String(filename))
end

"""
    import_onnx(filename; backend="auto") -> MLP

Load an MLP from an ONNX file and return a new instance.
"""
function import_onnx(filename::AbstractString; backend::String = "auto")
    ptr = c_import_onnx(String(filename), backend)
    MLP(ptr)
end

end # module
