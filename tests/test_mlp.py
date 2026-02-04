import pytest
from facaded_mlp_cuda import MLP, PyActivationType, PyOptimizerType

def test_create_mlp():
    mlp = MLP(2, [4], 1)
    assert mlp.input_size == 2
    assert mlp.output_size == 1
    assert mlp.hidden_sizes == [4]

def test_backends():
    backends = MLP.available_backends()
    assert "cpu" in backends
    
    # CPU should always work
    mlp = MLP(2, [4], 1, gpu_backend="cpu")
    assert mlp.gpu_backend == "cpu"

def test_xor_training():
    mlp = MLP(2, [8], 1, gpu_backend="cpu")
    mlp.learning_rate = 0.5
    
    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [[0], [1], [1], [0]]
    
    losses = mlp.fit(X, y, epochs=1000)
    
    # Loss should decrease
    assert losses[-1] < losses[0]
    
    # Predictions should be reasonable
    predictions = mlp.predict_batch(X)
    assert len(predictions) == 4
    assert all(0 <= p[0] <= 1 for p in predictions)

def test_save_load():
    mlp1 = MLP(3, [5, 5], 2, gpu_backend="cpu")
    mlp1.learning_rate = 0.123
    mlp1.save("/tmp/test_model.json")
    
    mlp2 = MLP.load("/tmp/test_model.json")
    assert mlp2.input_size == 3
    assert mlp2.output_size == 2
    assert mlp2.learning_rate == pytest.approx(0.123)

def test_properties():
    mlp = MLP(2, [4], 1, gpu_backend="cpu")
    
    mlp.learning_rate = 0.01
    assert mlp.learning_rate == 0.01
    
    mlp.optimizer = PyOptimizerType.Adam
    assert mlp.optimizer == PyOptimizerType.Adam
    
    mlp.dropout_rate = 0.5
    assert mlp.dropout_rate == 0.5

def test_backend_switching():
    mlp = MLP(2, [4], 1, gpu_backend="cpu")
    assert mlp.gpu_backend == "cpu"
    
    # Switching to CPU should always work
    mlp.set_backend("cpu")
    assert mlp.gpu_backend == "cpu"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
