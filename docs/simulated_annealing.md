# Simulated Annealing

## Overview

Simulated annealing (SA) is a probabilistic metaheuristic for global optimization, effective for combinatorial problems (e.g., TSP, scheduling) where the search space is large and gradient information is unavailable.

---

## Theory

### Metropolis Criterion

At each iteration, a neighbouring candidate solution is generated. If the candidate is better (lower cost) it is always accepted. If it is worse, it is accepted with probability:

```
P(accept) = exp(-(new_cost - current_cost) / T)
```

### Cooling Schedules

| Schedule | Formula | Behaviour |
|----------|---------|-----------|
| **Linear** | `T(i) = T₀ · max(0, 1 − i / max_iter)` | Uniform descent |
| **Exponential** | `T(i) = T₀ · αⁱ` | Geometric decay (most commonly used) |
| **Logarithmic** | `T(i) = T₀ / (1 + c · ln(1 + i))` | Slowest cooling |

---

## Python API

### `SimulatedAnnealing` class

```python
from python.simulated_annealing import SimulatedAnnealing, ExponentialCooling

sa = SimulatedAnnealing(
    objective_fn=my_cost,
    neighbor_fn=my_neighbor,
    initial_solution=start,
    initial_temp=1000.0,
    min_temp=1e-8,
    max_iterations=10_000,
    cooling_schedule=ExponentialCooling(alpha=0.995),
)
best_solution, best_cost, history = sa.optimize()
```

### Cooling Schedules

```python
from python.simulated_annealing.cooling_schedules import (
    LinearCooling,
    ExponentialCooling,
    LogarithmicCooling,
)

linear      = LinearCooling(max_iterations=10_000)
exponential = ExponentialCooling(alpha=0.995)
logarithmic = LogarithmicCooling(c=1.0)
```

### TSP example

```python
import math, random
from python.simulated_annealing import SimulatedAnnealing, ExponentialCooling

random.seed(42)
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)]

def tour_length(tour):
    return sum(
        math.dist(cities[tour[i]], cities[tour[(i+1) % len(tour)]])
        for i in range(len(tour))
    )

def two_opt(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    return tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]

sa = SimulatedAnnealing(
    objective_fn=tour_length,
    neighbor_fn=two_opt,
    initial_solution=list(range(len(cities))),
    initial_temp=5000.0,
    cooling_schedule=ExponentialCooling(alpha=0.9995),
)
best_tour, best_length, history = sa.optimize()
print(f"Optimized length: {best_length:.2f}")
```
