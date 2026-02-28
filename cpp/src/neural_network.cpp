#include "neural_network.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>

namespace ai {

// ─── Activation functions ────────────────────────────────────────────────────

namespace activation {

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double tanh_fn(double x) {
    return std::tanh(x);
}

double tanh_derivative(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

Vector softmax(const Vector& x) {
    Vector result(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    for (auto& v : result) v /= sum;
    return result;
}

} // namespace activation

// ─── Layer ───────────────────────────────────────────────────────────────────

Layer::Layer(int input_size, int output_size, const std::string& activation)
    : input_size_(input_size)
    , output_size_(output_size)
    , activation_name_(activation)
    , weights_(output_size, Vector(input_size, 0.0))
    , biases_(output_size, 0.0)
    , last_input_(input_size, 0.0)
    , last_output_(output_size, 0.0)
    , last_preactivation_(output_size, 0.0)
{
    if (activation == "relu") {
        activation_fn_    = activation::relu;
        activation_deriv_ = activation::relu_derivative;
    } else if (activation == "sigmoid") {
        activation_fn_    = activation::sigmoid;
        activation_deriv_ = activation::sigmoid_derivative;
    } else if (activation == "tanh") {
        activation_fn_    = activation::tanh_fn;
        activation_deriv_ = activation::tanh_derivative;
    } else {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
    initialize_weights();
}

void Layer::initialize_weights() {
    // He initialisation: std = sqrt(2 / fan_in)
    std::mt19937 rng(std::random_device{}());
    double std_dev = std::sqrt(2.0 / static_cast<double>(input_size_));
    std::normal_distribution<double> dist(0.0, std_dev);
    for (auto& row : weights_)
        for (auto& w : row)
            w = dist(rng);
}

Vector Layer::apply_activation(const Vector& x) const {
    Vector out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        out[i] = activation_fn_(x[i]);
    return out;
}

Vector Layer::apply_activation_derivative(const Vector& x) const {
    Vector out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        out[i] = activation_deriv_(x[i]);
    return out;
}

Vector Layer::forward(const Vector& input) {
    if (static_cast<int>(input.size()) != input_size_)
        throw std::invalid_argument("Input size mismatch in Layer::forward");

    last_input_ = input;

    // pre-activation: z = W * x + b
    Vector z(output_size_, 0.0);
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j)
            z[i] += weights_[i][j] * input[j];
        z[i] += biases_[i];
    }
    last_preactivation_ = z;
    last_output_ = apply_activation(z);
    return last_output_;
}

Vector Layer::backward(const Vector& grad_output, double learning_rate) {
    if (static_cast<int>(grad_output.size()) != output_size_)
        throw std::invalid_argument("Gradient size mismatch in Layer::backward");

    // delta = grad_output * activation'(z)
    Vector delta(output_size_);
    Vector act_deriv = apply_activation_derivative(last_preactivation_);
    for (int i = 0; i < output_size_; ++i)
        delta[i] = grad_output[i] * act_deriv[i];

    // gradient w.r.t. input: grad_input = W^T * delta
    Vector grad_input(input_size_, 0.0);
    for (int j = 0; j < input_size_; ++j)
        for (int i = 0; i < output_size_; ++i)
            grad_input[j] += weights_[i][j] * delta[i];

    // update weights and biases
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j)
            weights_[i][j] -= learning_rate * delta[i] * last_input_[j];
        biases_[i] -= learning_rate * delta[i];
    }

    return grad_input;
}

// ─── NeuralNetwork ───────────────────────────────────────────────────────────

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                             const std::vector<std::string>& activations,
                             double learning_rate)
    : learning_rate_(learning_rate)
{
    if (layer_sizes.size() < 2)
        throw std::invalid_argument("Network must have at least 2 layer sizes");

    std::size_t num_layers = layer_sizes.size() - 1;
    for (std::size_t i = 0; i < num_layers; ++i) {
        std::string act = "relu";
        if (!activations.empty()) {
            // last layer default sigmoid, hidden layers relu when not specified
            act = (i < activations.size()) ? activations[i] : "sigmoid";
        } else {
            act = (i == num_layers - 1) ? "sigmoid" : "relu";
        }
        layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1], act);
    }
}

Vector NeuralNetwork::forward(const Vector& input) {
    Vector x = input;
    for (auto& layer : layers_)
        x = layer.forward(x);
    return x;
}

void NeuralNetwork::backward(const Vector& target) {
    // Get last prediction from the stored outputs
    Vector prediction = layers_.back().last_output();
    Vector grad = mse_loss_gradient(prediction, target);
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
        grad = layers_[i].backward(grad, learning_rate_);
}

double NeuralNetwork::mse_loss(const Vector& prediction, const Vector& target) const {
    double loss = 0.0;
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        double diff = prediction[i] - target[i];
        loss += diff * diff;
    }
    return loss / static_cast<double>(prediction.size());
}

Vector NeuralNetwork::mse_loss_gradient(const Vector& prediction, const Vector& target) const {
    Vector grad(prediction.size());
    double n = static_cast<double>(prediction.size());
    for (std::size_t i = 0; i < prediction.size(); ++i)
        grad[i] = 2.0 * (prediction[i] - target[i]) / n;
    return grad;
}

void NeuralNetwork::train(const std::vector<Vector>& X, const std::vector<Vector>& y,
                          int epochs, int batch_size) {
    if (X.size() != y.size())
        throw std::invalid_argument("X and y must have the same number of samples");

    std::size_t n = X.size();
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;

        for (std::size_t start = 0; start < n; start += static_cast<std::size_t>(batch_size)) {
            std::size_t end = std::min(start + static_cast<std::size_t>(batch_size), n);
            for (std::size_t k = start; k < end; ++k) {
                Vector pred = forward(X[indices[k]]);
                epoch_loss += mse_loss(pred, y[indices[k]]);
                backward(y[indices[k]]);
            }
        }

        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch " << (epoch + 1)
                      << "  loss: " << epoch_loss / static_cast<double>(n) << "\n";
        }
    }
}

Vector NeuralNetwork::predict(const Vector& input) {
    return forward(input);
}

double NeuralNetwork::evaluate_accuracy(const std::vector<Vector>& X,
                                        const std::vector<int>& y_true) {
    if (X.size() != y_true.size())
        throw std::invalid_argument("X and y_true must have the same number of samples");

    int correct = 0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        Vector pred = predict(X[i]);
        int predicted = static_cast<int>(
            std::max_element(pred.begin(), pred.end()) - pred.begin());
        if (predicted == y_true[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(X.size());
}

void NeuralNetwork::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file for writing: " + filepath);

    file << layers_.size() << "\n";
    for (const auto& layer : layers_) {
        file << layer.input_size() << " " << layer.output_size() << "\n";
        for (const auto& row : layer.weights()) {
            for (std::size_t j = 0; j < row.size(); ++j)
                file << row[j] << (j + 1 < row.size() ? " " : "\n");
        }
        for (std::size_t j = 0; j < layer.biases().size(); ++j)
            file << layer.biases()[j] << (j + 1 < layer.biases().size() ? " " : "\n");
    }
}

void NeuralNetwork::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file for reading: " + filepath);

    std::size_t num_layers;
    file >> num_layers;
    if (num_layers != layers_.size())
        throw std::runtime_error("Layer count mismatch when loading model");

    for (auto& layer : layers_) {
        int in_sz, out_sz;
        file >> in_sz >> out_sz;
        if (in_sz != layer.input_size() || out_sz != layer.output_size())
            throw std::runtime_error("Layer dimension mismatch when loading model");

        // Load weights by casting away const via the mutable accessors
        // (weights/biases are logically part of model state; use const_cast here
        //  because the header exposes only const getters)
        auto& w = const_cast<Matrix&>(layer.weights());
        for (auto& row : w)
            for (auto& val : row)
                file >> val;

        auto& b = const_cast<Vector&>(layer.biases());
        for (auto& val : b)
            file >> val;
    }
}

} // namespace ai
