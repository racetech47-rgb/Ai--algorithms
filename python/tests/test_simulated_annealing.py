"""Unit tests for the simulated_annealing module."""

from __future__ import annotations

import math
import random
import unittest

from python.simulated_annealing.cooling_schedules import (
    ExponentialCooling,
    LinearCooling,
    LogarithmicCooling,
)
from python.simulated_annealing.annealing import SimulatedAnnealing


class TestCoolingSchedules(unittest.TestCase):
    """Tests for cooling schedule classes."""

    def _assert_decreasing(self, schedule, initial_temp: float = 100.0) -> None:
        temps = [schedule(initial_temp, i) for i in range(0, 1000, 50)]
        for a, b in zip(temps, temps[1:]):
            self.assertGreaterEqual(a, b, f"Temperature increased: {a} -> {b}")

    def _assert_non_negative(self, schedule, initial_temp: float = 100.0) -> None:
        for i in range(0, 500, 25):
            t = schedule(initial_temp, i)
            self.assertGreaterEqual(t, 0.0)

    # -- LinearCooling --

    def test_linear_initial_temp(self) -> None:
        sched = LinearCooling(max_iterations=1000)
        self.assertAlmostEqual(sched(100.0, 0), 100.0)

    def test_linear_reaches_zero(self) -> None:
        sched = LinearCooling(max_iterations=1000)
        self.assertAlmostEqual(sched(100.0, 1000), 0.0)

    def test_linear_decreasing(self) -> None:
        self._assert_decreasing(LinearCooling(max_iterations=1000))

    def test_linear_non_negative(self) -> None:
        self._assert_non_negative(LinearCooling(max_iterations=1000))

    def test_linear_invalid_max_iterations(self) -> None:
        with self.assertRaises(ValueError):
            LinearCooling(max_iterations=0)

    # -- ExponentialCooling --

    def test_exponential_initial_temp(self) -> None:
        sched = ExponentialCooling(alpha=0.99)
        self.assertAlmostEqual(sched(100.0, 0), 100.0)

    def test_exponential_decreasing(self) -> None:
        self._assert_decreasing(ExponentialCooling(alpha=0.99))

    def test_exponential_non_negative(self) -> None:
        self._assert_non_negative(ExponentialCooling(alpha=0.99))

    def test_exponential_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            ExponentialCooling(alpha=1.5)
        with self.assertRaises(ValueError):
            ExponentialCooling(alpha=0.0)

    # -- LogarithmicCooling --

    def test_logarithmic_initial_temp(self) -> None:
        sched = LogarithmicCooling(c=1.0)
        self.assertAlmostEqual(sched(100.0, 0), 100.0)

    def test_logarithmic_decreasing(self) -> None:
        self._assert_decreasing(LogarithmicCooling(c=1.0))

    def test_logarithmic_non_negative(self) -> None:
        self._assert_non_negative(LogarithmicCooling(c=1.0))

    def test_logarithmic_invalid_c(self) -> None:
        with self.assertRaises(ValueError):
            LogarithmicCooling(c=-1.0)


class TestSimulatedAnnealing(unittest.TestCase):
    """Tests for the SimulatedAnnealing optimizer."""

    def _quadratic_sa(self, initial: float = 10.0) -> SimulatedAnnealing:
        random.seed(42)
        return SimulatedAnnealing(
            objective_fn=lambda x: x ** 2,
            neighbor_fn=lambda x: x + random.uniform(-0.5, 0.5),
            initial_solution=initial,
            initial_temp=100.0,
            min_temp=1e-4,
            max_iterations=5000,
            cooling_schedule=ExponentialCooling(alpha=0.995),
        )

    # -- Return types --

    def test_returns_tuple_of_three(self) -> None:
        sa = self._quadratic_sa()
        result = sa.optimize()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_history_keys(self) -> None:
        _, _, history = self._quadratic_sa().optimize()
        for key in ("temperature", "cost", "best_cost"):
            self.assertIn(key, history)

    def test_history_lengths_match(self) -> None:
        _, _, history = self._quadratic_sa().optimize()
        lengths = {len(v) for v in history.values()}
        self.assertEqual(len(lengths), 1)

    # -- Optimization quality --

    def test_optimizes_quadratic(self) -> None:
        """Best solution should be near 0 (minimum of x^2)."""
        best_solution, best_cost, _ = self._quadratic_sa(initial=10.0).optimize()
        self.assertLess(best_cost, 1.0)

    def test_best_cost_non_increasing(self) -> None:
        _, _, history = self._quadratic_sa().optimize()
        best_costs = history["best_cost"]
        for a, b in zip(best_costs, best_costs[1:]):
            self.assertGreaterEqual(a, b - 1e-10)

    # -- Acceptance probability --

    def test_acceptance_always_1_for_improvement(self) -> None:
        p = SimulatedAnnealing.get_acceptance_probability(10.0, 5.0, 100.0)
        self.assertEqual(p, 1.0)

    def test_acceptance_between_0_and_1_for_worse(self) -> None:
        p = SimulatedAnnealing.get_acceptance_probability(5.0, 10.0, 100.0)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_acceptance_zero_at_zero_temp(self) -> None:
        p = SimulatedAnnealing.get_acceptance_probability(5.0, 10.0, 0.0)
        self.assertEqual(p, 0.0)

    # -- Stops at min_temp --

    def test_stops_before_max_iterations_when_cooled(self) -> None:
        sa = SimulatedAnnealing(
            objective_fn=lambda x: x ** 2,
            neighbor_fn=lambda x: x + random.uniform(-0.1, 0.1),
            initial_solution=5.0,
            initial_temp=1.0,
            min_temp=0.99,   # stops almost immediately
            max_iterations=10_000,
            cooling_schedule=ExponentialCooling(alpha=0.99),
        )
        _, _, history = sa.optimize()
        self.assertLess(len(history["cost"]), 10_000)


if __name__ == "__main__":
    unittest.main()
