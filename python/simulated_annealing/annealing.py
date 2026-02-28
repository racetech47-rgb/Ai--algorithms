"""Core simulated annealing optimizer."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cooling_schedules import CoolingSchedule, ExponentialCooling


class SimulatedAnnealing:
    """General-purpose simulated annealing optimizer.

    Args:
        objective_fn: Callable that takes a solution and returns its cost
            (lower is better).
        neighbor_fn: Callable that takes a solution and returns a
            neighbouring candidate solution.
        initial_solution: Starting point in the search space.
        initial_temp: Starting temperature (default 1000.0).
        min_temp: Minimum temperature; optimization stops when reached.
        max_iterations: Hard cap on the number of iterations.
        cooling_schedule: A :class:`CoolingSchedule` instance. Defaults
            to :class:`ExponentialCooling` with alpha=0.995.

    Example:
        >>> sa = SimulatedAnnealing(
        ...     objective_fn=lambda x: x**2,
        ...     neighbor_fn=lambda x: x + random.uniform(-0.5, 0.5),
        ...     initial_solution=10.0,
        ... )
        >>> solution, cost, history = sa.optimize()
    """

    def __init__(
        self,
        objective_fn: Callable[[Any], float],
        neighbor_fn: Callable[[Any], Any],
        initial_solution: Any,
        initial_temp: float = 1000.0,
        min_temp: float = 1e-8,
        max_iterations: int = 10_000,
        cooling_schedule: Optional[CoolingSchedule] = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.neighbor_fn = neighbor_fn
        self.initial_solution = initial_solution
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.cooling_schedule: CoolingSchedule = cooling_schedule or ExponentialCooling(
            alpha=0.995
        )

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_acceptance_probability(
        current_cost: float, new_cost: float, temp: float
    ) -> float:
        """Compute the Metropolis acceptance probability.

        Args:
            current_cost: Cost of the current solution.
            new_cost: Cost of the candidate solution.
            temp: Current temperature.

        Returns:
            Probability in [0, 1] of accepting the candidate solution.
        """
        if new_cost < current_cost:
            return 1.0
        if temp <= 0:
            return 0.0
        return math.exp(-(new_cost - current_cost) / temp)

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self) -> Tuple[Any, float, Dict[str, List[float]]]:
        """Run the simulated annealing algorithm.

        Returns:
            A tuple ``(best_solution, best_cost, history)`` where
            *history* is a dict with keys ``'temperature'``, ``'cost'``,
            and ``'best_cost'`` containing per-iteration values.
        """
        current_solution = self.initial_solution
        current_cost = self.objective_fn(current_solution)
        best_solution = current_solution
        best_cost = current_cost

        history: Dict[str, List[float]] = {
            "temperature": [],
            "cost": [],
            "best_cost": [],
        }

        for iteration in range(self.max_iterations):
            temp = self.cooling_schedule(self.initial_temp, iteration)

            if temp < self.min_temp:
                break

            candidate = self.neighbor_fn(current_solution)
            candidate_cost = self.objective_fn(candidate)

            acceptance = self.get_acceptance_probability(
                current_cost, candidate_cost, temp
            )
            if random.random() < acceptance:
                current_solution = candidate
                current_cost = candidate_cost

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            history["temperature"].append(temp)
            history["cost"].append(current_cost)
            history["best_cost"].append(best_cost)

        return best_solution, best_cost, history
