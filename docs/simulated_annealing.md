# Simulated Annealing

## Overview

Simulated annealing (SA) is a probabilistic metaheuristic for global optimization. It is especially effective for combinatorial problems (e.g., TSP, scheduling) where the search space is large and gradient information is unavailable. All three language implementations share the same algorithm structure and support pluggable cooling schedules.

---

## Theory

### Metropolis Criterion

At each iteration, a neighbouring candidate solution is generated. If the candidate is better (lower cost) it is always accepted. If it is worse, it is accepted with probability:

```
P(accept) = exp(-(new_cost - current_cost) / T)
```

where `T` is the current temperature. This allows the algorithm to escape local minima early in the search when `T` is high, gradually becoming more greedy as `T` falls.

### Cooling Schedules

The temperature `T` decreases from `initial_temp` toward `min_temp` according to a cooling schedule. Three schedules are provided:

| Schedule | Formula | Behaviour |
|----------|---------|-----------|
| **Linear** | `T(i) = T₀ · max(0, 1 − i / max_iter)` | Uniform descent; fast start, abrupt end |
| **Exponential** | `T(i) = T₀ · αⁱ` | Geometric decay; most commonly used |
| **Logarithmic** | `T(i) = T₀ / (1 + c · ln(1 + i))` | Slowest cooling; theoretically guarantees global optimum as iterations → ∞ |

---

## Python API

### `SimulatedAnnealing` class

```python
from python.simulated_annealing import SimulatedAnnealing, ExponentialCooling

sa = SimulatedAnnealing(
    objective_fn=my_cost,       # Callable[[Any], float]  — lower is better
    neighbor_fn=my_neighbor,    # Callable[[Any], Any]    — returns a new candidate
    initial_solution=start,     # Any                     — starting point
    initial_temp=1000.0,        # float
    min_temp=1e-8,              # float — stop when T drops below this
    max_iterations=10_000,      # int   — hard cap on iterations
    cooling_schedule=ExponentialCooling(alpha=0.995),
)
best_solution, best_cost, history = sa.optimize()
```

**`optimize()` return value**

| Key | Type | Description |
|-----|------|-------------|
| `best_solution` | Any | Best solution found |
| `best_cost` | float | Cost of the best solution |
| `history["temperature"]` | List[float] | Temperature at each iteration |
| `history["cost"]` | List[float] | Current solution cost at each iteration |
| `history["best_cost"]` | List[float] | Running best cost at each iteration |

**Static utility**

```python
p = SimulatedAnnealing.get_acceptance_probability(current_cost, new_cost, temp)
```

### Cooling Schedules

```python
from python.simulated_annealing.cooling_schedules import (
    LinearCooling,
    ExponentialCooling,
    LogarithmicCooling,
)

linear      = LinearCooling(max_iterations=10_000)
exponential = ExponentialCooling(alpha=0.995)   # alpha ∈ (0, 1)
logarithmic = LogarithmicCooling(c=1.0)         # c > 0
```

All schedules are callable: `schedule(initial_temp, iteration) → float`.

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

initial_tour = list(range(len(cities)))
random.shuffle(initial_tour)

sa = SimulatedAnnealing(
    objective_fn=tour_length,
    neighbor_fn=two_opt,
    initial_solution=initial_tour,
    initial_temp=5000.0,
    min_temp=1e-6,
    max_iterations=50_000,
    cooling_schedule=ExponentialCooling(alpha=0.9995),
)

best_tour, best_length, history = sa.optimize()
print(f"Initial tour length : {tour_length(initial_tour):.2f}")
print(f"Optimized length    : {best_length:.2f}")
print(f"Iterations run      : {len(history['cost'])}")
```

**Expected output**

```
Initial tour length : 842.37
Optimized length    : 423.15
Iterations run      : 49918
```

---

## C++ API

Header: `cpp/include/simulated_annealing.hpp`  
Namespace: `ai`

### `SimulatedAnnealing<Solution>` template class

```cpp
#include "simulated_annealing.hpp"

ai::CoolingSchedule cooling(ai::CoolingType::EXPONENTIAL, /*alpha=*/0.995);

ai::SimulatedAnnealing<double> sa(
    objective,          // std::function<double(const double&)>
    neighbor,           // std::function<double(const double&)>
    /*initial_temp=*/   100.0,
    /*min_temp=*/       1e-8,
    /*max_iterations=*/ 50000,
    cooling
);

auto [best, best_cost, history] = sa.optimize(initial_solution);
```

**`optimize()` return value** — `std::tuple<Solution, double, std::vector<AnnealingStep>>`

**`AnnealingStep` struct**

```cpp
struct AnnealingStep {
    int    iteration;
    double temperature;
    double current_cost;
    double best_cost;
};
```

History is recorded every 1000 iterations.

### `CoolingSchedule` class

```cpp
// CoolingType::LINEAR | EXPONENTIAL | LOGARITHMIC
ai::CoolingSchedule cooling(ai::CoolingType::EXPONENTIAL, /*alpha=*/0.995);
double temp = cooling(initial_temp, iteration);
```

### Static utility

```cpp
double p = ai::SimulatedAnnealing<double>::acceptance_probability(current_cost, new_cost, temp);
```

### Function minimization example

```cpp
#include "simulated_annealing.hpp"
#include <cmath>
#include <random>
#include <iostream>

int main() {
    // Minimize f(x) = (x - 3)²
    auto objective = [](const double& x) { return (x - 3.0) * (x - 3.0); };

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> perturb(0.0, 0.5);
    auto neighbor = [&](const double& x) { return x + perturb(rng); };

    ai::CoolingSchedule cooling(ai::CoolingType::EXPONENTIAL, 0.995);
    ai::SimulatedAnnealing<double> sa(objective, neighbor, 100.0, 1e-8, 50000, cooling);

    auto [best, best_cost, history] = sa.optimize(0.0);
    std::cout << "Best x    = " << best      << "\n";  // ≈ 3.0
    std::cout << "Best f(x) = " << best_cost << "\n";  // ≈ 0.0
}
```

---

## Java API

Package: `com.racetech.ai.annealing`

### `SimulatedAnnealing<S>` class

```java
import com.racetech.ai.annealing.SimulatedAnnealing;
import com.racetech.ai.annealing.ExponentialCooling;
import com.racetech.ai.annealing.AnnealingResult;

SimulatedAnnealing<List<Integer>> sa = new SimulatedAnnealing<>(
    TspExample::tourLength,      // Function<S, Double>  — objective
    TspExample::twoOptNeighbor,  // Function<S, S>       — neighbor generator
    /* initialTemp    */ 100.0,
    /* minTemp        */ 0.001,
    /* maxIterations  */ 10_000,
    new ExponentialCooling(0.9995)
);

AnnealingResult<List<Integer>> result = sa.optimize(initialTour);
```

**`AnnealingResult<S>`**

| Method | Return type | Description |
|--------|-------------|-------------|
| `getBestSolution()` | `S` | Best solution found |
| `getBestCost()` | `double` | Cost of best solution |
| `getCostHistory()` | `List<Double>` | Best cost recorded at each iteration |

### Cooling schedules

| Class | Constructor | Formula |
|-------|-------------|---------|
| `ExponentialCooling` | `ExponentialCooling(double alpha)` | `T₀ · αⁱ` |
| `LinearCooling` | `LinearCooling(int maxIterations)` | `T₀ · max(0, 1 − i/N)` |
| `LogarithmicCooling` | `LogarithmicCooling(double c)` | `T₀ / (1 + c·ln(1+i))` |

All implement `CoolingSchedule` via `computeTemperature(double initialTemp, int iteration)`.

### Static utility

```java
double p = SimulatedAnnealing.acceptanceProbability(currentCost, newCost, temp);
```

---

## Cooling Schedule Comparison

| Schedule | Speed | Exploration | Best for |
|----------|-------|-------------|----------|
| Linear | Fast | Low late-stage | Problems with known iteration budget |
| Exponential (`α=0.999`) | Medium | Balanced | General purpose (recommended default) |
| Exponential (`α=0.99`) | Fast | Lower | Quick runs, less accuracy needed |
| Logarithmic | Slow | High throughout | Problems requiring thorough search |

---

## Tuning Guide

### Choosing `initial_temp`

Set `initial_temp` so that roughly 80% of uphill moves are accepted at the start. A simple heuristic:

```python
import random, statistics

costs = [objective(neighbor(initial_solution)) for _ in range(100)]
delta_avg = statistics.mean(abs(c - objective(initial_solution)) for c in costs)
initial_temp = -delta_avg / math.log(0.8)
```

### Choosing `alpha` (exponential cooling)

| `alpha` | Iterations to cool 1000 → 1 |
|---------|------------------------------|
| 0.99 | ~688 |
| 0.995 | ~1379 |
| 0.999 | ~6906 |
| 0.9995 | ~13815 |

### Choosing `max_iterations`

A common rule of thumb: allow at least `10 × problem_size²` iterations for combinatorial problems.

### General recommendations

1. Start with `ExponentialCooling(alpha=0.995)` and `initial_temp=1000`.
2. If the algorithm converges too quickly to a poor solution, increase `initial_temp` or `alpha`.
3. If runtime is too long with little improvement, decrease `alpha` or `max_iterations`.
4. Use `history["best_cost"]` (Python) or `getCostHistory()` (Java) to plot convergence and diagnose premature convergence.
