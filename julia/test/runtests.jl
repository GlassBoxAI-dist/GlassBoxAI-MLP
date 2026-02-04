using Test
using FacadedMLP

@testset "FacadedMLP" begin
    @testset "MLP Creation" begin
        mlp = MLP(2, [8], 1)
        @test mlp.input_size == 2
        @test mlp.output_size == 1
        @test mlp.hidden_sizes == [8]
        @test mlp.num_layers == 3
    end
    
    @testset "MLP with Options" begin
        mlp = MLP(4, [16, 8], 2;
            hidden_activation = ReLU,
            output_activation = Softmax,
            learning_rate = 0.001,
            optimizer = Adam,
            dropout_rate = 0.1,
            l2_lambda = 0.0001
        )
        @test mlp.input_size == 4
        @test mlp.output_size == 2
        @test mlp.hidden_sizes == [16, 8]
        @test mlp.learning_rate ≈ 0.001
        @test mlp.optimizer == Adam
        @test mlp.dropout_rate ≈ 0.1
        @test mlp.l2_lambda ≈ 0.0001
    end
    
    @testset "Property Setters" begin
        mlp = MLP(2, [4], 1)
        
        mlp.learning_rate = 0.05
        @test mlp.learning_rate ≈ 0.05
        
        mlp.optimizer = SGD
        @test mlp.optimizer == SGD
        
        mlp.dropout_rate = 0.2
        @test mlp.dropout_rate ≈ 0.2
        
        mlp.batch_norm = true
        @test mlp.batch_norm == true
    end
    
    @testset "Predict" begin
        mlp = MLP(2, [4], 1)
        output = predict(mlp, [1.0, 0.0])
        @test length(output) == 1
        @test 0.0 <= output[1] <= 1.0
    end
    
    @testset "Train" begin
        mlp = MLP(2, [8], 1)
        train!(mlp, [1.0, 0.0], [1.0])
        output = predict(mlp, [1.0, 0.0])
        @test length(output) == 1
    end
    
    @testset "XOR Training" begin
        mlp = MLP(2, [8], 1;
            learning_rate = 0.5,
            optimizer = Adam
        )
        
        X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        y = [[0.0], [1.0], [1.0], [0.0]]
        
        losses = fit!(mlp, X, y; epochs=500)
        @test length(losses) == 500
        @test losses[end] < losses[1]
        
        predictions = predict_batch(mlp, X)
        @test length(predictions) == 4
    end
    
    @testset "Save and Load" begin
        mlp = MLP(2, [4], 1; learning_rate=0.123)
        train!(mlp, [1.0, 0.0], [1.0])
        
        filename = tempname() * ".json"
        save(mlp, filename)
        @test isfile(filename)
        
        mlp2 = load(filename)
        @test mlp2.input_size == 2
        @test mlp2.output_size == 1
        @test mlp2.learning_rate ≈ 0.123
        
        rm(filename)
    end
    
    @testset "Feature Importance" begin
        mlp = MLP(3, [4], 1)
        train!(mlp, [1.0, 0.5, 0.2], [1.0])
        
        importance = feature_importance(mlp)
        @test length(importance) == 3
        @test all(x -> x[1] >= 0 && x[1] < 3, importance)
    end
    
    @testset "Available Backends" begin
        backends = available_backends()
        @test "cpu" in backends
    end
    
    @testset "Neuron Access" begin
        mlp = MLP(2, [4], 1)
        
        weights = get_neuron_weights(mlp, 1, 0)
        @test length(weights) == 2
        
        bias = get_neuron_bias(mlp, 1, 0)
        @test typeof(bias) == Float64
        
        set_neuron_weight!(mlp, 1, 0, 0, 0.5)
        new_weights = get_neuron_weights(mlp, 1, 0)
        @test new_weights[1] ≈ 0.5
        
        set_neuron_bias!(mlp, 1, 0, -0.1)
        new_bias = get_neuron_bias(mlp, 1, 0)
        @test new_bias ≈ -0.1
    end
    
    @testset "Layer Outputs" begin
        mlp = MLP(2, [4], 1)
        predict(mlp, [1.0, 0.5])
        
        outputs = get_layer_outputs(mlp, 1)
        @test length(outputs) == 4
    end
end

println("All tests passed!")
