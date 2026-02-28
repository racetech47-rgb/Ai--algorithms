# Problem Solver

## Overview

`ProblemSolver` is a high-level facade (Python only) that routes diverse AI tasks to the appropriate sub-module. Rather than wiring up preprocessors, models, and optimizers yourself, you describe *what* you want to solve and pass the relevant data.

```python
from python.problem_solver import ProblemSolver

ps = ProblemSolver()
result = ps.auto_solve("sentiment", texts=["I love this!"])
```

---

## How Components Are Combined

```
ProblemSolver.auto_solve(problem_type, **kwargs)
         │
         ├─ "classification" → TextPreprocessor → TextClassifier
         │                      (nlp.preprocessing) (nlp.text_classification)
         │
         ├─ "optimization"   → SimulatedAnnealing + ExponentialCooling
         │                      (simulated_annealing.annealing)
         │
         └─ "sentiment"      → SentimentAnalyzer
                                (nlp.sentiment_analysis)
```

All sub-modules are lazily imported, so missing optional dependencies (e.g., `transformers`) only raise an error if the relevant solver is actually called.

---

## Usage Examples

### Sentiment Analysis

```python
from python.problem_solver import ProblemSolver

ps = ProblemSolver()
result = ps.auto_solve(
    "sentiment",
    texts=[
        "I am so happy with this purchase!",
        "This is the worst product I have ever bought.",
        "It is okay, nothing extraordinary.",
    ]
)

for text, r in zip(texts, result["results"]):
    print(f"[{r['label']} {r['score']:.3f}]")

print(result["summary"])
# {"POSITIVE": 2, "NEGATIVE": 1}
```

**Return value**

| Key | Type | Description |
|-----|------|-------------|
| `results` | `List[dict]` | Per-text `{"label": str, "score": float}` |
| `summary` | `dict` | Count of each label across all texts |

---

### Text Classification

```python
from python.problem_solver import ProblemSolver

ps = ProblemSolver()

train_texts = [
    "The movie was fantastic and I loved every minute",
    "Terrible film, total waste of time",
    "Absolutely brilliant performance by the cast",
    "Boring and predictable, fell asleep halfway through",
    "Outstanding special effects and gripping storyline",
    "Dull script, poor acting, not worth watching",
    "A masterpiece of modern cinema",
    "Disappointing sequel that ruins the original",
]
train_labels = [1, 0, 1, 0, 1, 0, 1, 0]

test_texts = [
    "Wonderful experience, would watch again",
    "Dreadful movie, avoid at all costs",
]

result = ps.auto_solve(
    "classification",
    texts=train_texts,
    labels=train_labels,
    test_texts=test_texts,
)

print(result["predictions"])     # [1, 0]
print(result["probabilities"])   # [[0.11, 0.89], [0.94, 0.06]]
print(result["metrics"])
# {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
```

**`solve_classification` parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `List[str]` | Training texts |
| `labels` | `List[int]` | Integer class labels for training samples |
| `test_texts` | `List[str]` | Texts to classify |

**Return value**

| Key | Type | Description |
|-----|------|-------------|
| `predictions` | `List[int]` | Predicted integer labels for test texts |
| `probabilities` | `List[List[float]]` | Class probability estimates (n_test × n_classes) |
| `metrics` | `dict` | Training-set metrics: `accuracy`, `precision`, `recall`, `f1` |

---

### TSP Optimization

```python
from python.problem_solver import ProblemSolver

ps = ProblemSolver()

# Option 1: provide your own city coordinates
cities = [(0, 0), (3, 4), (6, 1), (5, 7), (2, 8)]
result = ps.auto_solve("optimization", cities=cities)

# Option 2: generate random cities
result = ps.auto_solve("optimization", num_cities=15)

print(f"Initial length  : {result['initial_length']:.2f}")
print(f"Optimized length: {result['best_length']:.2f}")
print(f"Improvement     : {result['improvement_pct']:.1f}%")
print(f"Iterations      : {result['iterations']}")
print(f"Best tour       : {result['best_tour']}")
```

**`solve_optimization` parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cities` | `List[Tuple[float, float]]` | `None` | City (x, y) coordinates; if `None`, random cities are generated |
| `num_cities` | `int` | `15` | Number of random cities to generate when `cities` is `None` |

**Return value**

| Key | Type | Description |
|-----|------|-------------|
| `best_tour` | `List[int]` | Optimized city visit order |
| `best_length` | `float` | Total Euclidean tour length after optimization |
| `initial_length` | `float` | Tour length before optimization |
| `improvement_pct` | `float` | Percentage improvement over the initial tour |
| `iterations` | `int` | Number of SA iterations performed |

SA settings used internally: `initial_temp=5000`, `min_temp=1e-6`, `max_iterations=30_000`, `ExponentialCooling(alpha=0.9995)`.

---

## Direct Method Calls

You can call the individual solver methods directly instead of routing through `auto_solve`:

```python
ps = ProblemSolver()

# Equivalent to auto_solve("sentiment", texts=[...])
result = ps.solve_nlp_task(texts=["Great product!"])

# Equivalent to auto_solve("classification", texts=..., labels=..., test_texts=...)
result = ps.solve_classification(texts, labels, test_texts)

# Equivalent to auto_solve("optimization", num_cities=20)
result = ps.solve_optimization(num_cities=20)
```

---

## Integration Patterns

### Running the full integrated example

```bash
cd /path/to/Ai--algorithms
python -m python.problem_solver.examples.integrated_example
```

**Output**

```
==================================================
TEXT CLASSIFICATION
==================================================
Predictions : [1, 0]
Train metrics:
  accuracy    : 1.0000
  precision   : 1.0000
  recall      : 1.0000
  f1          : 1.0000

==================================================
TSP OPTIMIZATION (15 random cities)
==================================================
Initial tour length  : 612.34
Optimized tour length : 398.71
Improvement          : 34.9%
Iterations           : 29987

==================================================
SENTIMENT ANALYSIS
==================================================
[POSITIVE 0.950]  I am so happy with this purchase!
[NEGATIVE 0.990]  This is the worst product I have ever bought.
[POSITIVE 0.512]  It is okay, nothing extraordinary.
Summary: {'POSITIVE': 2, 'NEGATIVE': 1}
```

### Chaining preprocessing with classification

If you want finer control over preprocessing before classification, use the sub-modules directly and pass processed text to `solve_classification`:

```python
from python.nlp.preprocessing import TextPreprocessor
from python.problem_solver import ProblemSolver

tp = TextPreprocessor(remove_stopwords=True, lemmatization=True)
processed_train = tp.batch_preprocess(raw_train_texts)
processed_test  = tp.batch_preprocess(raw_test_texts)

ps = ProblemSolver()
result = ps.solve_classification(
    texts=processed_train,
    labels=train_labels,
    test_texts=processed_test,
)
```

### Using a custom cooling schedule with the optimization solver

The `solve_optimization` method uses hard-coded SA parameters. For full control, use `SimulatedAnnealing` directly (see [`docs/simulated_annealing.md`](simulated_annealing.md)) and then pass your city list to the solver as coordinates only.
