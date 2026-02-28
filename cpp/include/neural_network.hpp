#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace ai {

// Type aliases
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Activation functions
namespace activation {
    double relu(double x);
    double relu_derivative(double x);
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double tanh_fn(double x);
    double tanh_derivative(double x);
    Vector softmax(const Vector& x);
}

// Layer class
class Layer {
public:
    Layer(int input_size, int output_size, const std::string& activation = "relu");

    Vector forward(const Vector& input);
    Vector backward(const Vector& grad_output, double learning_rate);

    const Matrix& weights() const { return weights_; }
    const Vector& biases() const { return biases_; }
    const Vector& last_output() const { return last_output_; }
    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }

private:
    int input_size_;
    int output_size_;
    std::string activation_name_;
    Matrix weights_;
    Vector biases_;
    Vector last_input_;
    Vector last_output_;
    Vector last_preactivation_;

    std::function<double(double)> activation_fn_;
    std::function<double(double)> activation_deriv_;

    void initialize_weights();
    Vector apply_activation(const Vector& x) const;
    Vector apply_activation_derivative(const Vector& x) const;
};

// Neural Network class
class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes,
                  const std::vector<std::string>& activations = {},
                  double learning_rate = 0.01);

    Vector forward(const Vector& input);
    void backward(const Vector& target);
    void train(const std::vector<Vector>& X, const std::vector<Vector>& y,
               int epochs = 100, int batch_size = 32);

    Vector predict(const Vector& input);
    double evaluate_accuracy(const std::vector<Vector>& X, const std::vector<int>& y_true);

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    std::vector<Layer> layers_;
    double learning_rate_;

    double mse_loss(const Vector& prediction, const Vector& target) const;
    Vector mse_loss_gradient(const Vector& prediction, const Vector& target) const;
};

} // namespace ai
