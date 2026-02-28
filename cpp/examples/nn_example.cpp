#include "neural_network.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // XOR training data
    std::vector<ai::Vector> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<ai::Vector> y = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Network: 2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid)
    ai::NeuralNetwork nn({2, 4, 1}, {"tanh", "sigmoid"}, /*learning_rate=*/0.5);

    std::cout << "Training neural network on XOR problem...\n\n";
    nn.train(X, y, /*epochs=*/1000, /*batch_size=*/4);

    std::cout << "\nPredictions after training:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < X.size(); ++i) {
        ai::Vector pred = nn.predict(X[i]);
        std::cout << "  [" << X[i][0] << ", " << X[i][1] << "]"
                  << "  =>  " << pred[0]
                  << "  (expected " << y[i][0] << ")\n";
    }

    return 0;
}
