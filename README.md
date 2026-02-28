# AI Algorithms Suite

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

A comprehensive suite of production-quality AI algorithm implementations covering neural networks, simulated annealing, natural language processing, and an integrated problem solver.

---

## Features

- **Multi-Layer Perceptron (MLP)** — configurable feed-forward neural network with ReLU / Sigmoid / Tanh activations, dropout, and Adam/SGD optimizers
- **Simulated Annealing** — generic optimizer with pluggable cooling schedules (linear, exponential, logarithmic) and Metropolis acceptance
- **NLP Pipeline** — text preprocessing (tokenization, stop-word removal, lemmatization), transformer-backed sentiment analysis with lexicon fallback, and TF-IDF + Logistic Regression text classification
- **Integrated Problem Solver** — high-level facade that routes classification, optimization, and sentiment tasks to the appropriate sub-module
- Fully documented public APIs with type annotations
- Unit-tested with pytest

---

## Project Structure

```
Ai--algorithms/
├── python/
│   ├── neural_network/          # MLP implementation (PyTorch)
│   │   ├── mlp.py               #   MLP model class
│   │   └── trainer.py           #   Training / evaluation utilities
│   ├── simulated_annealing/     # SA optimizer
│   │   ├── annealing.py         #   SimulatedAnnealing class
│   │   └── cooling_schedules.py #   Linear / Exponential / Logarithmic schedules
│   ├── nlp/                     # NLP components
│   │   ├── preprocessing.py     #   TextPreprocessor
│   │   ├── sentiment_analysis.py#   SentimentAnalyzer
│   │   └── text_classification.py#  TextClassifier
│   ├── problem_solver/          # Integrated facade
│   │   └── solver.py            #   ProblemSolver class
│   ├── tests/                   # pytest test suite
│   └── requirements.txt
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

---

## Installation

```bash
cd python
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `scikit-learn>=1.3`, `nltk>=3.8`, `transformers>=4.30`.

---

## Quick Start

### Neural Network (Iris classification)

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

### Simulated Annealing (TSP)

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

### NLP Sentiment Analysis

```python
from python.nlp import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("I love this product!")
print(result)  # {'label': 'POSITIVE', 'score': 0.99...}
```

---

## Testing

```bash
cd python
pip install -r requirements.txt
pytest tests/ -v
```

---

## API Reference

Full API documentation lives in the `docs/` directory:

- [`docs/neural_network.md`](docs/neural_network.md)
- [`docs/simulated_annealing.md`](docs/simulated_annealing.md)
- [`docs/nlp.md`](docs/nlp.md)
- [`docs/problem_solver.md`](docs/problem_solver.md)

---

## License

This project is licensed under the **MIT License**.
