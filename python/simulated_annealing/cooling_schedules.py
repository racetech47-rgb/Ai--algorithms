"""Cooling schedule strategies for simulated annealing."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class CoolingSchedule(ABC):
    """Abstract base class for cooling schedules.

    Subclasses implement ``__call__`` to return the temperature at a
    given iteration given the initial temperature.
    """

    @abstractmethod
    def __call__(self, initial_temp: float, iteration: int) -> float:
        """Return temperature for *iteration* starting from *initial_temp*.

        Args:
            initial_temp: Starting temperature (> 0).
            iteration: Current iteration index (0-based).

        Returns:
            Temperature value for the given iteration.
        """


class LinearCooling(CoolingSchedule):
    """Linearly decreasing temperature schedule.

    T(i) = initial_temp * max(0, 1 - i / max_iterations)

    Args:
        max_iterations: Total number of iterations; determines the slope.
    """

    def __init__(self, max_iterations: int = 10_000) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        self.max_iterations = max_iterations

    def __call__(self, initial_temp: float, iteration: int) -> float:  # noqa: D401
        fraction = max(0.0, 1.0 - iteration / self.max_iterations)
        return initial_temp * fraction


class ExponentialCooling(CoolingSchedule):
    """Exponentially decreasing temperature schedule.

    T(i) = initial_temp * alpha^i

    Args:
        alpha: Cooling rate in (0, 1). Smaller values cool faster.
    """

    def __init__(self, alpha: float = 0.995) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = alpha

    def __call__(self, initial_temp: float, iteration: int) -> float:  # noqa: D401
        return initial_temp * (self.alpha ** iteration)


class LogarithmicCooling(CoolingSchedule):
    """Logarithmically decreasing temperature schedule.

    T(i) = initial_temp / (1 + c * log(1 + i))

    Args:
        c: Cooling constant (default 1.0). Larger values cool faster.
    """

    def __init__(self, c: float = 1.0) -> None:
        if c <= 0:
            raise ValueError("c must be positive.")
        self.c = c

    def __call__(self, initial_temp: float, iteration: int) -> float:  # noqa: D401
        return initial_temp / (1.0 + self.c * math.log(1.0 + iteration))
