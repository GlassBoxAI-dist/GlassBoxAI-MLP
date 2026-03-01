# Mutation API Progress

## Goal
Add mutation (set) methods for every introspection (get) method in the facade,
then propagate to CLI, and all language wrappers.

## New Mutation APIs Needed

| Property | Get (exists) | Set (needed) |
|---|---|---|
| Weight M (Adam) | `get_weight_m` | `set_weight_m` |
| Weight V (Adam) | `get_weight_v` | `set_weight_v` |
| Bias M (Adam) | `get_bias_m` | `set_bias_m` |
| Bias V (Adam) | `get_bias_v` | `set_bias_v` |
| Timestep | `timestep` | `set_timestep` |
| Layer Activation | `get_layer_activation` | `set_layer_activation` |
| All Weights (bulk) | `get_weights` | `set_weights` |

### CLI Commands Needed
- `set-weight-m`, `set-weight-v`, `set-bias-m`, `set-bias-v`
- `set-weights` (bulk)
- `set-activation` (layer activation)
- `set-timestep`
- `set-learning-rate`, `set-optimizer-type`, `set-dropout`, `set-l2`, `set-batch-norm`

## Progress

### Phase 1: Core Rust (mlp.rs)
- [x] `SetWeightM(layer, neuron, weight_idx, value)`
- [x] `SetWeightV(layer, neuron, weight_idx, value)`
- [x] `SetBiasM(layer, neuron, value)`
- [x] `SetBiasV(layer, neuron, value)`
- [x] `SetTimestep(value)`
- [x] `SetLayerActivation(layer, activation)`
- [x] `SetNeuronWeights(layer, neuron, weights)`

### Phase 2: Facade (facade.rs)
- [x] `set_weight_m`, `set_weight_v`, `set_bias_m`, `set_bias_v`
- [x] `set_timestep`
- [x] `set_layer_activation`
- [x] `set_weights` (bulk)

### Phase 3: CLI (cli.rs)
- [ ] `set-weight-m`, `set-weight-v`, `set-bias-m`, `set-bias-v`
- [ ] `set-weights`
- [ ] `set-activation`
- [ ] `set-timestep`
- [ ] `set-learning-rate`, `set-optimizer-type`, `set-dropout`, `set-l2`, `set-batch-norm`
- [ ] Update --help text

### Phase 4: C FFI (julia.rs) + C Header (facaded_mlp.h)
- [x] `mlp_set_weight_m`, `mlp_set_weight_v`, `mlp_set_bias_m`, `mlp_set_bias_v`
- [x] `mlp_set_timestep`
- [x] `mlp_set_layer_activation`
- [x] `mlp_set_neuron_weights`
- [x] Update facaded_mlp.h

### Phase 5: C++ Wrapper (facaded_mlp.hpp)
- [x] All new set methods

### Phase 6: C# Wrapper (NativeMethods.cs + MLP.cs)
- [x] P/Invoke declarations + managed wrappers

### Phase 7: Go Wrapper (mlp.go)
- [x] CGo declarations + Go methods

### Phase 8: Julia Wrapper (FacadedMLP.jl)
- [x] ccall wrappers + public functions

### Phase 9: Python Wrapper (python.rs)
- [x] PyO3 methods

### Phase 10: Node.js Wrapper (nodejs.rs + index.d.ts)
- [x] NAPI methods + TypeScript declarations

### Phase 11: Zig Wrapper (c.zig + mlp.zig)
- [x] C declarations + Zig methods
