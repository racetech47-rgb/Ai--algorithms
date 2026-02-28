# AI Algorithms Suite

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![Java](https://img.shields.io/badge/Java-11%2B-blue?logo=openjdk)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

A multi-language suite of production-quality AI algorithm implementations covering neural networks, simulated annealing, natural language processing, and an integrated problem solver. Each algorithm is implemented independently in Python, C++, and Java so you can use whichever runtime best fits your project.

---

## Features

- **Multi-Layer Perceptron (MLP)** — configurable feed-forward neural network with ReLU / Sigmoid / Tanh / Softmax activations, dropout, and Adam/SGD optimizers
- **Simulated Annealing** — generic optimizer with pluggable cooling schedules (linear, exponential, logarithmic) and Metropolis acceptance
- **NLP Pipeline** — text preprocessing (tokenization, stop-word removal, lemmatization), transformer-backed sentiment analysis with lexicon fallback, and TF-IDF + Logistic Regression text classification
- **Integrated Problem Solver** — high-level facade that routes classification, optimization, and sentiment tasks to the appropriate sub-module
- Fully documented public APIs with type annotations (Python) / Javadoc (Java) / Doxygen-compatible comments (C++)
- Unit-tested in all three languages

---

## Project Structure

```
Ai--algorithms/
├── python/
│   ├── neural_network/          # MLP implementation (PyTorch)
│   │   ├── mlp.py               #   MLP model class
│   │   ├── trainer.py           #   Training / evaluation utilities
│   │   └── examples/
│   │       └── iris_classification.py
│   ├── simulated_annealing/     # SA optimizer
│   │   ├── annealing.py         #   SimulatedAnnealing class
│   │   ├── cooling_schedules.py #   Linear / Exponential / Logarithmic schedules
│   │   └── examples/
│   │       └── tsp_solver.py
│   ├── nlp/                     # NLP components
│   │   ├── preprocessing.py     #   TextPreprocessor
│   │   ├── sentiment_analysis.py#   SentimentAnalyzer
│   │   ├── text_classification.py#  TextClassifier
│   │   └── examples/
│   │       └── sentiment_example.py
│   ├── problem_solver/          # Integrated facade
│   │   ├── solver.py            #   ProblemSolver class
│   │   └── examples/
│   │       └── integrated_example.py
│   ├── tests/                   # pytest test suite
│   └── requirements.txt
├── cpp/
│   ├── include/
│   │   ├── neural_network.hpp   # NeuralNetwork / Layer API
│   │   └── simulated_annealing.hpp # SimulatedAnnealing<T> template
│   ├── src/
│   │   ├── neural_network.cpp
│   │   └── simulated_annealing.cpp
│   ├── examples/
│   │   ├── nn_example.cpp       # XOR demo
│   │   └── sa_example.cpp       # f(x)=(x-3)² minimization demo
│   └── CMakeLists.txt
├── java/
│   ├── src/main/java/com/racetech/ai/
│   │   ├── neuralnetwork/       # NeuralNetwork, Layer, Trainer, Activation
│   │   └── annealing/           # SimulatedAnnealing<S>, cooling schedules
│   ├── src/test/java/           # JUnit test suite
│   └── pom.xml
└── docs/
    ├── neural_network.md
    ├── simulated_annealing.md
    ├── nlp.md
    └── problem_solver.md
```

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.9 |
| pip | 21.0 |
| GCC / Clang | C++17 support |
| CMake | 3.15 |
| Java JDK | 11 |
| Maven | 3.8 |

---

## Installation

### Python

```bash
cd python
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `scikit-learn>=1.3`, `nltk>=3.8`, `transformers>=4.30`.

### C++

```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

This produces two static libraries (`libneural_network.a`, `libsimulated_annealing.a`) and two demo binaries (`nn_example`, `sa_example`).

### Java

```bash
cd java
mvn clean install
```

The Maven build compiles all sources, runs the JUnit test suite, and packages a JAR.

---

## Quick Start

### Python — Neural Network (Iris classification)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from python.neural_network import MLP, Trainer

iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

model   = MLP(input_size=4, hidden_sizes=[64, 32], output_size=3, activation="relu", dropout=0.2)
trainer = Trainer(model, learning_rate=0.001)
history = trainer.train(X_train, y_train, epochs=100, batch_size=16,
                        validation_data=(X_test, y_test))

metrics = trainer.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

### Python — Simulated Annealing (TSP)

```python
import math, random
from python.simulated_annealing import SimulatedAnnealing, ExponentialCooling

random.seed(42)
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)]

def tour_length(tour):
    return sum(math.dist(cities[tour[i]], cities[tour[(i+1) % len(tour)]])
               for i in range(len(tour)))

def two_opt(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    return tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]

sa = SimulatedAnnealing(
    objective_fn=tour_length,
    neighbor_fn=two_opt,
    initial_solution=list(range(20)),
    initial_temp=5000.0,
    cooling_schedule=ExponentialCooling(alpha=0.9995),
)
best_tour, best_length, history = sa.optimize()
print(f"Optimized tour length: {best_length:.2f}")
```

### C++ — Neural Network (XOR)

```cpp
#include "neural_network.hpp"
#include <iostream>

int main() {
    // 2 inputs → 4 hidden (tanh) → 1 output (sigmoid)
    ai::NeuralNetwork nn({2, 4, 1}, {"tanh", "sigmoid"}, /*lr=*/0.5);

    std::vector<ai::Vector> X = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<ai::Vector> y = {{0},{1},{1},{0}};
    nn.train(X, y, /*epochs=*/1000);

    for (std::size_t i = 0; i < X.size(); ++i)
        std::cout << nn.predict(X[i])[0] << "\n";
}
```

### Java — Neural Network (Iris)

```java
NeuralNetwork network = new NeuralNetwork(
    new int[]        {4, 8, 3},
    new Activation[] {Activation.RELU, Activation.SOFTMAX},
    0.01
);
Trainer trainer = new Trainer(network);
trainer.train(X, yOneHot, 100, 5);
double accuracy = network.evaluateAccuracy(X, yTrue);
System.out.printf("Accuracy: %.1f%%%n", accuracy * 100);
```

---

## Architecture Overview

| Module | Language | Description |
|--------|----------|-------------|
| `neural_network/mlp.py` | Python | PyTorch `nn.Module` MLP with configurable depth, activations, and dropout |
| `neural_network/trainer.py` | Python | Training loop, evaluation (accuracy, precision, recall, F1), inference |
| `cpp/include/neural_network.hpp` | C++ | Hand-rolled MLP with Xavier init, MSE loss, and SGD backprop |
| `java/.../NeuralNetwork.java` | Java | Feed-forward net with MSE loss, supports mini-batch SGD |
| `simulated_annealing/annealing.py` | Python | Generic SA optimizer; accepts any objective/neighbor callable |
| `simulated_annealing/cooling_schedules.py` | Python | `LinearCooling`, `ExponentialCooling`, `LogarithmicCooling` |
| `cpp/include/simulated_annealing.hpp` | C++ | Template class `SimulatedAnnealing<Solution>` |
| `java/.../SimulatedAnnealing.java` | Java | Generic `SimulatedAnnealing<S>` with pluggable `CoolingSchedule` |
| `nlp/preprocessing.py` | Python | `TextPreprocessor` — tokenize, stopword removal, lemmatize/stem |
| `nlp/sentiment_analysis.py` | Python | `SentimentAnalyzer` — transformer pipeline with lexicon fallback |
| `nlp/text_classification.py` | Python | `TextClassifier` — TF-IDF + Logistic Regression pipeline |
| `problem_solver/solver.py` | Python | `ProblemSolver` facade — routes to classification, optimization, or sentiment |

---

## API Reference Overview

Full API documentation lives in the `docs/` directory:

- [`docs/neural_network.md`](docs/neural_network.md) — MLP classes, Trainer, Layer in all three languages
- [`docs/simulated_annealing.md`](docs/simulated_annealing.md) — SA optimizer, cooling schedules, tuning guide
- [`docs/nlp.md`](docs/nlp.md) — preprocessing pipeline, sentiment analysis, text classification
- [`docs/problem_solver.md`](docs/problem_solver.md) — integrated facade and integration patterns

---

## Testing

### Python

```bash
cd python
pytest tests/ -v
```

### C++

```bash
cd cpp/build
ctest --output-on-failure
```

### Java

```bash
cd java
mvn test
```

---

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Write code and accompanying tests.
3. Ensure all existing tests pass (`pytest`, `ctest`, `mvn test`).
4. Open a pull request describing your changes.

Please follow PEP 8 for Python, the Google C++ style guide for C++, and Google Java Style for Java.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
