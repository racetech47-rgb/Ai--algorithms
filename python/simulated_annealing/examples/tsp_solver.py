"""Travelling Salesman Problem (TSP) solver using simulated annealing."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.simulated_annealing import SimulatedAnnealing, ExponentialCooling


# ── TSP helpers ─────────────────────────────────────────────────────────────

def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def tour_length(tour: List[int], cities: List[Tuple[float, float]]) -> float:
    """Return the total Euclidean length of a city tour (closed loop).

    Args:
        tour: Ordered list of city indices.
        cities: List of (x, y) coordinates.

    Returns:
        Total tour length.
    """
    total = sum(
        _euclidean(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )
    return total


def two_opt_neighbor(tour: List[int]) -> List[int]:
    """Generate a neighbour by performing a random 2-opt swap.

    Args:
        tour: Current tour as a list of city indices.

    Returns:
        New tour with one 2-opt move applied.
    """
    n = len(tour)
    i, j = sorted(random.sample(range(n), 2))
    new_tour = tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]
    return new_tour


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate random cities, solve TSP, and print results."""
    random.seed(42)
    num_cities = 20
    cities: List[Tuple[float, float]] = [
        (random.uniform(0, 100), random.uniform(0, 100))
        for _ in range(num_cities)
    ]

    initial_tour = list(range(num_cities))
    random.shuffle(initial_tour)

    objective = lambda tour: tour_length(tour, cities)  # noqa: E731
    neighbor = two_opt_neighbor
    sa = SimulatedAnnealing(
        objective_fn=objective,
        neighbor_fn=neighbor,
        initial_solution=initial_tour,
        initial_temp=5000.0,
        min_temp=1e-6,
        max_iterations=50_000,
        cooling_schedule=ExponentialCooling(alpha=0.9995),
    )

    print(f"Initial tour length : {tour_length(initial_tour, cities):.2f}")
    best_tour, best_length, history = sa.optimize()
    print(f"Optimized tour length : {best_length:.2f}")
    print(f"Iterations run : {len(history['cost'])}")
    print(f"Best tour : {best_tour}")


if __name__ == "__main__":
    main()
