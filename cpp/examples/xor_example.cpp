/**
 * @file
 * @ingroup MLP_Internal_Logic
 */
/*
 * XOR Example - Demonstrates basic MLP usage
 *
 * Build:
 *   g++ -std=c++17 -O2 -I../include xor_example.cpp -L../../target/release -lfacaded_mlp_cuda -o xor_example
 *
 * Run:
 *   LD_LIBRARY_PATH=../../target/release ./xor_example
 */

#include <iostream>
#include <iomanip>
#include "facaded_mlp.hpp"

int main() {
    using namespace facaded;
    
    std::cout << "=== XOR Example ===" << std::endl;
    std::cout << "Available backends: ";
    for (const auto& b : MLP::available_backends()) {
        std::cout << b << " ";
    }
    std::cout << std::endl << std::endl;
    
    // Create network: 2 inputs, 8 hidden neurons, 1 output
    MLPOptions opts;
    opts.hidden_activation = Activation::Sigmoid;
    opts.output_activation = Activation::Sigmoid;
    opts.learning_rate = 0.5;
    opts.optimizer = Optimizer::Adam;
    opts.backend = "auto";
    
    MLP mlp(2, {8}, 1, opts);
    std::cout << "Created: " << mlp << std::endl << std::endl;
    
    // XOR training data
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    // Train
    std::cout << "Training..." << std::endl;
    auto result = mlp.fit(inputs, targets, 1000, true);
    std::cout << std::endl;
    
    // Test predictions
    std::cout << "Predictions:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = mlp.predict(inputs[i]);
        std::cout << "  [" << inputs[i][0] << ", " << inputs[i][1] << "] => "
                  << output[0] << " (expected: " << targets[i][0] << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Save and reload
    std::cout << "Saving model to xor_model.json..." << std::endl;
    mlp.save("xor_model.json");
    
    std::cout << "Loading model..." << std::endl;
    auto mlp2 = MLP::load("xor_model.json");
    std::cout << "Loaded: " << mlp2 << std::endl;
    
    // Verify loaded model
    std::cout << "\nVerifying loaded model:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = mlp2.predict(inputs[i]);
        std::cout << "  [" << inputs[i][0] << ", " << inputs[i][1] << "] => "
                  << output[0] << std::endl;
    }
    
    // Feature importance
    std::cout << "\nFeature importance:" << std::endl;
    auto importance = mlp.feature_importance();
    for (const auto& fi : importance) {
        std::cout << "  Feature " << fi.index << ": " << fi.score << std::endl;
    }
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

