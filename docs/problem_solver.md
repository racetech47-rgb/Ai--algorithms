# Problem Solver

## Overview

`ProblemSolver` is a high-level facade that routes diverse AI tasks to the appropriate sub-module.

```python
from python.problem_solver import ProblemSolver

ps = ProblemSolver()
result = ps.auto_solve("sentiment", texts=["I love this!"])
```

---

## Problem Types

### Sentiment Analysis

```python
result = ps.auto_solve("sentiment", texts=["I love this!", "Terrible product."])
# result["results"]  → [{"label": "POSITIVE", ...}, {"label": "NEGATIVE", ...}]
# result["summary"]  → {"POSITIVE": 1, "NEGATIVE": 1}
```

### Text Classification

```python
result = ps.auto_solve(
    "classification",
    texts=train_texts,
    labels=train_labels,
    test_texts=test_texts,
)
# result["predictions"]   → [1, 0, ...]
# result["probabilities"] → [[0.1, 0.9], ...]
# result["metrics"]       → {"accuracy": ..., "precision": ..., ...}
```

### TSP Optimization

```python
result = ps.auto_solve("optimization", num_cities=15)
# result["best_tour"]       → [0, 3, 1, ...]
# result["best_length"]     → 398.71
# result["initial_length"]  → 612.34
# result["improvement_pct"] → 34.9
# result["iterations"]      → 29987
```

---

## Direct Method Calls

```python
ps.solve_nlp_task(texts=[...])
ps.solve_classification(texts, labels, test_texts)
ps.solve_optimization(cities=None, num_cities=15)
```
