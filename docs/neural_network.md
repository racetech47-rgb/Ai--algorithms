# Neural Network

## Overview

The neural network module provides a configurable Multi-Layer Perceptron (MLP) implemented independently in Python (PyTorch), C++ (hand-rolled SGD), and Java (mini-batch SGD). All three share the same conceptual API: construct a network by specifying layer sizes and activations, train it on labelled data, and call predict/evaluate on unseen samples.

---

## Architecture

Each implementation follows the same MLP design:

```
Input Layer  →  [Hidden Layer(s)]  →  Output Layer
    ↓                  ↓                   ↓
 n_features      Linear + Activation    n_classes
```

- **Weight initialization** — Xavier / Glorot uniform (C++, Java), PyTorch default (Python)
- **Loss function** — Cross-Entropy (Python), MSE (C++, Java)
- **Activations** — ReLU, Sigmoid, Tanh, Softmax
- **Optimizers** — Adam or SGD (Python), SGD (C++, Java)
- **Regularization** — Dropout (Python)

---

## Python API

### `MLP` class

```python
from python.neural_network import MLP

model = MLP(
    input_size=4,        # number of input features
    hidden_sizes=[64, 32],  # list of hidden layer widths
    output_size=3,       # number of output classes
    activation="relu",   # "relu" | "sigmoid" | "tanh"
    dropout=0.2,         # dropout probability in [0, 1)
)
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(x: Tensor) → Tensor` | Forward pass; returns raw logits |
| `save` | `(path: str) → None` | Saves state dict to disk |
| `load` | `(path, input_size, hidden_sizes, output_size, ...) → MLP` | Loads weights from disk (classmethod) |

### `Trainer` class

```python
from python.neural_network import MLP, Trainer

model   = MLP(4, [64, 32], 3)
trainer = Trainer(model, learning_rate=0.001, optimizer="adam")
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `train` | `(X_train, y_train, epochs, batch_size, validation_data) → dict` | Trains the model; returns loss/accuracy history |
| `evaluate` | `(X_test, y_test) → dict` | Returns `loss`, `accuracy`, `precision`, `recall`, `f1` |
| `predict` | `(X) → np.ndarray` | Returns integer class predictions |

#### Training example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from python.neural_network import MLP, Trainer

iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

model   = MLP(input_size=4, hidden_sizes=[64, 32], output_size=3, activation="relu", dropout=0.2)
trainer = Trainer(model, learning_rate=0.001)

history = trainer.train(X_train, y_train, epochs=100, batch_size=16,
                        validation_data=(X_test, y_test))
print(f"Final train acc: {history['train_acc'][-1]:.4f}")
print(f"Final val   acc: {history['val_acc'][-1]:.4f}")

metrics = trainer.evaluate(X_test, y_test)
# {'loss': 0.0821, 'accuracy': 0.9667, 'precision': 0.9677, 'recall': 0.9667, 'f1': 0.9667}
```

#### Saving and loading

```python
model.save("iris_mlp.pt")

loaded = MLP.load("iris_mlp.pt", input_size=4, hidden_sizes=[64, 32], output_size=3)
preds  = trainer.predict(X_test)  # numpy array of class indices
```

---

## C++ API

Header: `cpp/include/neural_network.hpp`  
Namespace: `ai`

### `NeuralNetwork` class

```cpp
#include "neural_network.hpp"

// layer_sizes: [input, hidden..., output]
// activations: one string per layer ("relu", "sigmoid", "tanh")
ai::NeuralNetwork nn({4, 64, 32, 3}, {"relu", "relu", "sigmoid"}, /*learning_rate=*/0.01);
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(const Vector& input) → Vector` | Forward pass through all layers |
| `backward` | `(const Vector& target) → void` | Backpropagation with MSE loss |
| `train` | `(X, y, epochs, batch_size) → void` | Full training loop |
| `predict` | `(const Vector& input) → Vector` | Forward pass (alias) |
| `evaluate_accuracy` | `(X, y_true) → double` | Classification accuracy |
| `save` | `(filepath) → void` | Saves weights to file |
| `load` | `(filepath) → void` | Loads weights from file |

**Type aliases**

```cpp
using ai::Vector = std::vector<double>;
using ai::Matrix = std::vector<std::vector<double>>;
```

#### XOR example

```cpp
#include "neural_network.hpp"
#include <iostream>

int main() {
    ai::NeuralNetwork nn({2, 4, 1}, {"tanh", "sigmoid"}, 0.5);

    std::vector<ai::Vector> X = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<ai::Vector> y = {{0},{1},{1},{0}};
    nn.train(X, y, /*epochs=*/1000, /*batch_size=*/4);

    for (std::size_t i = 0; i < X.size(); ++i) {
        auto pred = nn.predict(X[i]);
        std::cout << "[" << X[i][0] << "," << X[i][1] << "] => " << pred[0] << "\n";
    }
}
```

Expected output:
```
[0,0] => 0.0312
[0,1] => 0.9701
[1,0] => 0.9698
[1,1] => 0.0287
```

### `Layer` class

```cpp
ai::Layer layer(input_size, output_size, "relu");
ai::Vector out  = layer.forward(input);
ai::Vector grad = layer.backward(grad_output, learning_rate);
```

---

## Java API

Package: `com.racetech.ai.neuralnetwork`

### `NeuralNetwork` class

```java
import com.racetech.ai.neuralnetwork.Activation;
import com.racetech.ai.neuralnetwork.NeuralNetwork;

// layerSizes: [inputSize, hidden..., outputSize]
NeuralNetwork network = new NeuralNetwork(
    new int[]        {4, 8, 3},
    new Activation[] {Activation.RELU, Activation.SOFTMAX},
    /*learningRate=*/ 0.01
);
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(double[] input) → double[]` | Forward pass; returns output activations |
| `backward` | `(double[] target) → void` | Backpropagation with MSE loss |
| `predict` | `(double[] input) → int` | Returns argmax class index |
| `evaluateAccuracy` | `(double[][] X, int[] yTrue) → double` | Fraction correctly classified |
| `computeLoss` | `(double[] target) → double` | MSE loss on last forward output |
| `getLayers` | `() → List<Layer>` | Unmodifiable layer list |

### `Trainer` class

```java
import com.racetech.ai.neuralnetwork.Trainer;

Trainer trainer = new Trainer(network);
Map<String, List<Double>> history = trainer.train(X, yOneHot, 100, 5);
Map<String, Double> metrics = trainer.evaluate(X, yTrue);
```

**`train` return value** — map with key `"loss"` containing per-epoch loss values.

**`evaluate` return value** — map with keys `"accuracy"`, `"precision"`, `"recall"`, `"f1"`.

### `Activation` enum

| Constant | Description |
|----------|-------------|
| `RELU` | Rectified Linear Unit |
| `SIGMOID` | Logistic sigmoid |
| `TANH` | Hyperbolic tangent |
| `SOFTMAX` | Softmax (use on output layer for multi-class) |

#### Iris example

```java
NeuralNetwork network = new NeuralNetwork(
    new int[]        {4, 8, 3},
    new Activation[] {Activation.RELU, Activation.SOFTMAX},
    0.01
);
Trainer trainer = new Trainer(network);
Map<String, List<Double>> history = trainer.train(X, yOneHot, 100, 5);

Map<String, Double> metrics = trainer.evaluate(X, yTrue);
System.out.printf("Accuracy: %.1f%%%n", metrics.get("accuracy") * 100);

int pred = network.predict(new double[]{0.222, 0.625, 0.068, 0.042});
System.out.println("Predicted class: " + pred);  // 0 (setosa)
```

---

## Configuration Options

| Option | Python | C++ | Java | Default | Notes |
|--------|--------|-----|------|---------|-------|
| Layer sizes | `hidden_sizes` list | `layer_sizes` vector | `layerSizes` int[] | — | Required |
| Activation | `activation` str | `activations` vector | `Activation[]` | `"relu"` / `RELU` | Per-layer in C++/Java |
| Learning rate | `learning_rate` | constructor arg | constructor arg | `0.001` / `0.01` | |
| Dropout | `dropout` float | — | — | `0.0` | Python only |
| Optimizer | `optimizer` str | SGD only | SGD only | `"adam"` | Python: adam or sgd |
| Batch size | `batch_size` | `batch_size` | `batchSize` | `32` / `5` | |
| Epochs | `epochs` | `epochs` | `epochs` | `50` / `100` | |

---

## Example Output

```
Training MLP on Iris dataset …
  Final train acc : 0.9833 | val acc : 0.9667

Test-set metrics:
  loss        : 0.0821
  accuracy    : 0.9667
  precision   : 0.9677
  recall      : 0.9667
  f1          : 0.9667
```
