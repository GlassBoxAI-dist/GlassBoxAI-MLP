/**
 * @file
 * @ingroup MLP_Internal_Logic
 */
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * C++ wrapper for Facaded MLP CUDA/OpenCL library
 */

#ifndef FACADED_MLP_HPP
#define FACADED_MLP_HPP

#include "facaded_mlp.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <cmath>

namespace facaded {

// Activation function types
enum class Activation {
    Sigmoid = MLP_ACTIVATION_SIGMOID,
    Tanh = MLP_ACTIVATION_TANH,
    ReLU = MLP_ACTIVATION_RELU,
    Softmax = MLP_ACTIVATION_SOFTMAX
};

// Optimizer types
enum class Optimizer {
    SGD = MLP_OPTIMIZER_SGD,
    Adam = MLP_OPTIMIZER_ADAM,
    RMSProp = MLP_OPTIMIZER_RMSPROP
};

// Exception class
class MLPException : public std::runtime_error {
public:
    explicit MLPException(const std::string& msg) : std::runtime_error(msg) {}
};

// Training result
struct TrainResult {
    std::vector<double> losses;
    double final_loss;
};

// Feature importance entry
struct FeatureImportance {
    int index;
    double score;
};

// MLP configuration options
struct MLPOptions {
    Activation hidden_activation = Activation::Sigmoid;
    Activation output_activation = Activation::Sigmoid;
    std::string backend = "auto";
    double learning_rate = 0.01;
    Optimizer optimizer = Optimizer::Adam;
    double dropout_rate = 0.0;
    double l2_lambda = 0.0;
    bool batch_norm = false;
};

/**
 * GPU-accelerated Multi-Layer Perceptron
 *
 * Example:
 * @code
 * facaded::MLP mlp(2, {8}, 1);
 * mlp.fit({{0,0}, {0,1}, {1,0}, {1,1}}, {{0}, {1}, {1}, {0}}, 1000);
 * auto output = mlp.predict({1.0, 0.0});
 * @endcode
 */
class MLP {
public:
    /**
     * Create a new MLP model
     *
     * @param input_size Number of input neurons
     * @param hidden_sizes List of hidden layer sizes
     * @param output_size Number of output neurons
     * @param options Configuration options
     */
    MLP(int input_size, const std::vector<int>& hidden_sizes, int output_size,
        const MLPOptions& options = MLPOptions())
        : input_size_(input_size), output_size_(output_size), hidden_sizes_(hidden_sizes)
    {
        std::vector<int32_t> hidden(hidden_sizes.begin(), hidden_sizes.end());
        
        handle_ = mlp_create(
            static_cast<int32_t>(input_size),
            hidden.data(),
            static_cast<int32_t>(hidden.size()),
            static_cast<int32_t>(output_size),
            static_cast<int32_t>(options.hidden_activation),
            static_cast<int32_t>(options.output_activation),
            options.backend.c_str()
        );
        
        if (!handle_) {
            const char* err = mlp_get_last_error();
            throw MLPException(err ? err : "Failed to create MLP");
        }
        
        set_learning_rate(options.learning_rate);
        set_optimizer(options.optimizer);
        set_dropout_rate(options.dropout_rate);
        set_l2_lambda(options.l2_lambda);
        set_batch_norm(options.batch_norm);
    }
    
    // Disable copy
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    
    // Enable move
    MLP(MLP&& other) noexcept
        : handle_(other.handle_), input_size_(other.input_size_),
          output_size_(other.output_size_), hidden_sizes_(std::move(other.hidden_sizes_))
    {
        other.handle_ = nullptr;
    }
    
    MLP& operator=(MLP&& other) noexcept {
        if (this != &other) {
            if (handle_) mlp_destroy(handle_);
            handle_ = other.handle_;
            input_size_ = other.input_size_;
            output_size_ = other.output_size_;
            hidden_sizes_ = std::move(other.hidden_sizes_);
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    ~MLP() {
        if (handle_) {
            mlp_destroy(handle_);
        }
    }
    
    /**
     * Load model from file
     */
    static MLP load(const std::string& filename) {
        mlp_handle_t handle = mlp_load(filename.c_str());
        if (!handle) {
            const char* err = mlp_get_last_error();
            throw MLPException(err ? err : "Failed to load model");
        }
        return MLP(handle);
    }
    
    /**
     * Train on a single sample
     */
    void train(const std::vector<double>& input, const std::vector<double>& target) {
        int32_t status = mlp_train(
            handle_,
            input.data(), static_cast<int32_t>(input.size()),
            target.data(), static_cast<int32_t>(target.size())
        );
        if (status != 0) {
            const char* err = mlp_get_last_error();
            throw MLPException(err ? err : "Training failed");
        }
    }
    
    /**
     * Make a prediction
     */
    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> output(output_size_);
        int32_t len = mlp_predict(
            handle_,
            input.data(), static_cast<int32_t>(input.size()),
            output.data(), static_cast<int32_t>(output.size())
        );
        if (len < 0) {
            const char* err = mlp_get_last_error();
            throw MLPException(err ? err : "Prediction failed");
        }
        output.resize(len);
        return output;
    }
    
    /**
     * Train on a dataset for multiple epochs
     *
     * @param inputs Vector of input samples
     * @param targets Vector of target outputs
     * @param epochs Number of training epochs
     * @param verbose Print progress every 100 epochs
     * @return Training result with loss history
     */
    TrainResult fit(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets,
                    int epochs = 100, bool verbose = false)
    {
        if (inputs.size() != targets.size()) {
            throw MLPException("inputs and targets must have same length");
        }
        
        TrainResult result;
        result.losses.reserve(epochs);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            
            for (size_t i = 0; i < inputs.size(); ++i) {
                train(inputs[i], targets[i]);
                auto output = predict(inputs[i]);
                epoch_loss += compute_loss(output, targets[i]);
            }
            
            epoch_loss /= static_cast<double>(inputs.size());
            result.losses.push_back(epoch_loss);
            
            if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                          << " - Loss: " << epoch_loss << std::endl;
            }
        }
        
        result.final_loss = result.losses.empty() ? 0.0 : result.losses.back();
        return result;
    }
    
    /**
     * Predict on multiple samples
     */
    std::vector<std::vector<double>> predict_batch(const std::vector<std::vector<double>>& inputs) {
        std::vector<std::vector<double>> outputs;
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            outputs.push_back(predict(input));
        }
        return outputs;
    }
    
    /**
     * Compute loss for given output and target
     */
    double compute_loss(const std::vector<double>& output, const std::vector<double>& target) {
        return mlp_compute_loss(
            handle_,
            output.data(), static_cast<int32_t>(output.size()),
            target.data(), static_cast<int32_t>(target.size())
        );
    }
    
    /**
     * Save model to file
     */
    void save(const std::string& filename) const {
        int32_t status = mlp_save(handle_, filename.c_str());
        if (status != 0) {
            const char* err = mlp_get_last_error();
            throw MLPException(err ? err : "Save failed");
        }
    }
    
    /**
     * Compute feature importance
     */
    std::vector<FeatureImportance> feature_importance() const {
        std::vector<int32_t> indices(input_size_);
        std::vector<double> scores(input_size_);
        
        int32_t len = mlp_feature_importance(
            handle_,
            indices.data(), scores.data(),
            static_cast<int32_t>(input_size_)
        );
        
        std::vector<FeatureImportance> result;
        result.reserve(len);
        for (int32_t i = 0; i < len; ++i) {
            result.push_back({static_cast<int>(indices[i]), scores[i]});
        }
        return result;
    }
    
    // Property getters
    double learning_rate() const { return mlp_get_learning_rate(handle_); }
    Optimizer optimizer() const { return static_cast<Optimizer>(mlp_get_optimizer(handle_)); }
    double dropout_rate() const { return mlp_get_dropout_rate(handle_); }
    double l2_lambda() const { return mlp_get_l2_lambda(handle_); }
    bool batch_norm() const { return mlp_get_batch_norm(handle_) != 0; }
    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }
    const std::vector<int>& hidden_sizes() const { return hidden_sizes_; }
    int num_layers() const { return mlp_get_num_layers(handle_); }
    
    std::string backend() const {
        const char* b = mlp_get_backend(handle_);
        return b ? b : "unknown";
    }
    
    // Property setters
    void set_learning_rate(double value) { mlp_set_learning_rate(handle_, value); }
    void set_optimizer(Optimizer value) { mlp_set_optimizer(handle_, static_cast<int32_t>(value)); }
    void set_dropout_rate(double value) { mlp_set_dropout_rate(handle_, value); }
    void set_l2_lambda(double value) { mlp_set_l2_lambda(handle_, value); }
    void set_batch_norm(bool value) { mlp_set_batch_norm(handle_, value ? 1 : 0); }
    
    void set_backend(const std::string& backend) {
        int32_t status = mlp_set_backend(handle_, backend.c_str());
        if (status != 0) {
            const char* err = mlp_get_last_error();
            throw MLPException(err ? err : "Set backend failed");
        }
    }
    
    /**
     * Get available GPU backends
     */
    static std::vector<std::string> available_backends() {
        char* backends = mlp_available_backends();
        if (!backends) return {"cpu"};
        
        std::vector<std::string> result;
        std::string s(backends);
        mlp_free_string(backends);
        
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            result.push_back(item);
        }
        return result;
    }
    
    /**
     * Get weights for a specific neuron
     */
    std::vector<double> get_neuron_weights(int layer, int neuron) const {
        int prev_size = (layer == 1) ? input_size_ : 
                        (layer <= static_cast<int>(hidden_sizes_.size())) ? hidden_sizes_[layer - 2] : 
                        hidden_sizes_.back();
        
        std::vector<double> weights(prev_size);
        int32_t len = mlp_get_neuron_weights(
            handle_, layer, neuron,
            weights.data(), static_cast<int32_t>(prev_size)
        );
        weights.resize(len);
        return weights;
    }
    
    /**
     * Get bias for a specific neuron
     */
    double get_neuron_bias(int layer, int neuron) const {
        return mlp_get_neuron_bias(handle_, layer, neuron);
    }
    
    /**
     * Set a specific weight
     */
    void set_neuron_weight(int layer, int neuron, int weight_idx, double value) {
        mlp_set_neuron_weight(handle_, layer, neuron, weight_idx, value);
    }
    
    /**
     * Set a neuron's bias
     */
    void set_neuron_bias(int layer, int neuron, double value) {
        mlp_set_neuron_bias(handle_, layer, neuron, value);
    }
    
    /**
     * Get layer outputs (after prediction)
     */
    std::vector<double> get_layer_outputs(int layer) {
        int layer_size;
        if (layer == 0) {
            layer_size = input_size_;
        } else if (layer <= static_cast<int>(hidden_sizes_.size())) {
            layer_size = hidden_sizes_[layer - 1];
        } else {
            layer_size = output_size_;
        }
        
        std::vector<double> outputs(layer_size);
        int32_t len = mlp_get_layer_outputs(
            handle_, layer,
            outputs.data(), static_cast<int32_t>(layer_size)
        );
        outputs.resize(len);
        return outputs;
    }
    
    /**
     * Get layer errors/gradients (after training)
     */
    std::vector<double> get_layer_errors(int layer) {
        int layer_size;
        if (layer == 0) {
            layer_size = input_size_;
        } else if (layer <= static_cast<int>(hidden_sizes_.size())) {
            layer_size = hidden_sizes_[layer - 1];
        } else {
            layer_size = output_size_;
        }

        std::vector<double> errors(layer_size);
        int32_t len = mlp_get_layer_errors(
            handle_, layer,
            errors.data(), static_cast<int32_t>(layer_size)
        );
        errors.resize(len);
        return errors;
    }

    /**
     * Get the size of a layer
     */
    int get_layer_size(int layer) const {
        return mlp_get_layer_size(handle_, layer);
    }

    /**
     * Get the activation type of a layer
     */
    Activation get_layer_activation(int layer) const {
        return static_cast<Activation>(mlp_get_layer_activation(handle_, layer));
    }

    /**
     * Get Adam optimizer's first moment (M) for a weight
     */
    double get_weight_m(int layer, int neuron, int weight_idx) const {
        return mlp_get_weight_m(handle_, layer, neuron, weight_idx);
    }

    /**
     * Get Adam optimizer's second moment (V) for a weight
     */
    double get_weight_v(int layer, int neuron, int weight_idx) const {
        return mlp_get_weight_v(handle_, layer, neuron, weight_idx);
    }

    /**
     * Get Adam optimizer's first moment (M) for a bias
     */
    double get_bias_m(int layer, int neuron) const {
        return mlp_get_bias_m(handle_, layer, neuron);
    }

    /**
     * Get Adam optimizer's second moment (V) for a bias
     */
    double get_bias_v(int layer, int neuron) const {
        return mlp_get_bias_v(handle_, layer, neuron);
    }

    /**
     * Get activation histogram for a layer
     */
    std::vector<int> get_activation_histogram(int layer, int bins) const {
        std::vector<int32_t> hist(bins);
        int32_t len = mlp_get_activation_histogram(
            handle_, layer, bins,
            hist.data(), static_cast<int32_t>(bins)
        );
        return std::vector<int>(hist.begin(), hist.begin() + len);
    }

    /**
     * Get gradient histogram for a layer
     */
    std::vector<int> get_gradient_histogram(int layer, int bins) const {
        std::vector<int32_t> hist(bins);
        int32_t len = mlp_get_gradient_histogram(
            handle_, layer, bins,
            hist.data(), static_cast<int32_t>(bins)
        );
        return std::vector<int>(hist.begin(), hist.begin() + len);
    }

    /**
     * Get Adam optimizer timestep
     */
    int timestep() const {
        return mlp_get_timestep(handle_);
    }

    /**
     * String representation
     */
    std::string to_string() const {
        std::stringstream ss;
        ss << "MLP(input=" << input_size_ << ", hidden=[";
        for (size_t i = 0; i < hidden_sizes_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << hidden_sizes_[i];
        }
        ss << "], output=" << output_size_
           << ", lr=" << learning_rate()
           << ", optimizer=" << optimizer_to_string(optimizer())
           << ", backend=" << backend() << ")";
        return ss.str();
    }

private:
    mlp_handle_t handle_;
    int input_size_;
    int output_size_;
    std::vector<int> hidden_sizes_;
    
    // Private constructor for load()
    explicit MLP(mlp_handle_t handle) : handle_(handle) {
        input_size_ = mlp_get_input_size(handle);
        output_size_ = mlp_get_output_size(handle);
        
        std::vector<int32_t> sizes(100);
        int32_t count = mlp_get_hidden_sizes(handle, sizes.data(), 100);
        hidden_sizes_.assign(sizes.begin(), sizes.begin() + count);
    }
    
    static const char* optimizer_to_string(Optimizer opt) {
        switch (opt) {
            case Optimizer::SGD: return "SGD";
            case Optimizer::Adam: return "Adam";
            case Optimizer::RMSProp: return "RMSProp";
            default: return "Unknown";
        }
    }
};

// Stream operator
inline std::ostream& operator<<(std::ostream& os, const MLP& mlp) {
    return os << mlp.to_string();
}

} // namespace facaded

#endif // FACADED_MLP_HPP
