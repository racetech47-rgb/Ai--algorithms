# Neural Network

## Overview

The neural network module provides a configurable Multi-Layer Perceptron (MLP) implemented in Python using PyTorch. Construct a network by specifying layer sizes and activations, train it on labelled data, and call predict/evaluate on unseen samples.

---

## Architecture

```
Input Layer  →  [Hidden Layer(s)]  →  Output Layer
    ↓                  ↓                   ↓
 n_features      Linear + Activation    n_classes
```

- **Loss function** — Cross-Entropy
- **Activations** — ReLU, Sigmoid, Tanh
- **Optimizers** — Adam or SGD
- **Regularization** — Dropout

---

## Python API

### `MLP` class

```python
from python.neural_network import MLP

model = MLP(
    input_size=4,
    hidden_sizes=[64, 32],
    output_size=3,
    activation="relu",   # "relu" | "sigmoid" | "tanh"
    dropout=0.2,
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
metrics = trainer.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```
