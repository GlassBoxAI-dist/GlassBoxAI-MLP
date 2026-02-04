package facadedmlp

import (
	"os"
	"testing"
)

func TestNewMLP(t *testing.T) {
	mlp, err := New(2, []int{4}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	if mlp.InputSize() != 2 {
		t.Errorf("Expected input size 2, got %d", mlp.InputSize())
	}
	if mlp.OutputSize() != 1 {
		t.Errorf("Expected output size 1, got %d", mlp.OutputSize())
	}
	if len(mlp.HiddenSizes()) != 1 || mlp.HiddenSizes()[0] != 4 {
		t.Errorf("Expected hidden sizes [4], got %v", mlp.HiddenSizes())
	}
}

func TestNewMLPWithConfig(t *testing.T) {
	config := &Config{
		HiddenActivation: ReLU,
		OutputActivation: Sigmoid,
		Backend:          BackendCPU,
		LearningRate:     0.001,
		Optimizer:        Adam,
		DropoutRate:      0.1,
		L2Lambda:         0.0001,
	}

	mlp, err := New(4, []int{16, 8}, 2, config)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	if mlp.LearningRate() != 0.001 {
		t.Errorf("Expected learning rate 0.001, got %f", mlp.LearningRate())
	}
	if mlp.Optimizer() != Adam {
		t.Errorf("Expected optimizer Adam, got %v", mlp.Optimizer())
	}
	if mlp.DropoutRate() != 0.1 {
		t.Errorf("Expected dropout rate 0.1, got %f", mlp.DropoutRate())
	}
}

func TestPredict(t *testing.T) {
	mlp, err := New(2, []int{4}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	output, err := mlp.Predict([]float64{1.0, 0.0})
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	if len(output) != 1 {
		t.Errorf("Expected 1 output, got %d", len(output))
	}
	if output[0] < 0 || output[0] > 1 {
		t.Errorf("Expected output in [0,1], got %f", output[0])
	}
}

func TestTrain(t *testing.T) {
	mlp, err := New(2, []int{4}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	err = mlp.Train([]float64{1.0, 0.0}, []float64{1.0})
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}
}

func TestXORTraining(t *testing.T) {
	config := &Config{
		HiddenActivation: Sigmoid,
		OutputActivation: Sigmoid,
		Backend:          BackendCPU,
		LearningRate:     0.5,
		Optimizer:        Adam,
	}

	mlp, err := New(2, []int{8}, 1, config)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	targets := [][]float64{{0}, {1}, {1}, {0}}

	result, err := mlp.Fit(inputs, targets, 500, false)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if len(result.Losses) != 500 {
		t.Errorf("Expected 500 losses, got %d", len(result.Losses))
	}

	// Loss should decrease
	if result.FinalLoss >= result.Losses[0] {
		t.Errorf("Loss should decrease: initial=%f, final=%f", result.Losses[0], result.FinalLoss)
	}
}

func TestSaveLoad(t *testing.T) {
	mlp, err := New(2, []int{4}, 1, &Config{LearningRate: 0.123})
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}

	// Train once
	mlp.Train([]float64{1.0, 0.0}, []float64{1.0})

	filename := "test_model.json"
	defer os.Remove(filename)

	err = mlp.Save(filename)
	if err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	mlp.Close()

	mlp2, err := Load(filename)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer mlp2.Close()

	if mlp2.InputSize() != 2 {
		t.Errorf("Loaded model has wrong input size: %d", mlp2.InputSize())
	}
	if mlp2.LearningRate() != 0.123 {
		t.Errorf("Loaded model has wrong learning rate: %f", mlp2.LearningRate())
	}
}

func TestFeatureImportance(t *testing.T) {
	mlp, err := New(3, []int{4}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	mlp.Train([]float64{1.0, 0.5, 0.2}, []float64{1.0})

	importance := mlp.FeatureImportance()
	if len(importance) != 3 {
		t.Errorf("Expected 3 features, got %d", len(importance))
	}
}

func TestNeuronAccess(t *testing.T) {
	mlp, err := New(2, []int{4}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	weights := mlp.GetNeuronWeights(1, 0)
	if len(weights) != 2 {
		t.Errorf("Expected 2 weights, got %d", len(weights))
	}

	bias := mlp.GetNeuronBias(1, 0)
	_ = bias // Just checking it doesn't crash

	mlp.SetNeuronWeight(1, 0, 0, 0.5)
	newWeights := mlp.GetNeuronWeights(1, 0)
	if newWeights[0] != 0.5 {
		t.Errorf("Expected weight 0.5, got %f", newWeights[0])
	}

	mlp.SetNeuronBias(1, 0, -0.1)
	newBias := mlp.GetNeuronBias(1, 0)
	if newBias != -0.1 {
		t.Errorf("Expected bias -0.1, got %f", newBias)
	}
}

func TestLayerOutputs(t *testing.T) {
	mlp, err := New(2, []int{4}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	mlp.Predict([]float64{1.0, 0.5})

	outputs := mlp.GetLayerOutputs(1)
	if len(outputs) != 4 {
		t.Errorf("Expected 4 outputs, got %d", len(outputs))
	}
}

func TestAvailableBackends(t *testing.T) {
	backends := AvailableBackends()
	if len(backends) == 0 {
		t.Error("Expected at least one backend")
	}

	hasCPU := false
	for _, b := range backends {
		if b == "cpu" {
			hasCPU = true
			break
		}
	}
	if !hasCPU {
		t.Error("Expected CPU backend to be available")
	}
}

func TestString(t *testing.T) {
	mlp, err := New(2, []int{8}, 1, nil)
	if err != nil {
		t.Fatalf("Failed to create MLP: %v", err)
	}
	defer mlp.Close()

	s := mlp.String()
	if s == "" {
		t.Error("String() returned empty string")
	}
}
