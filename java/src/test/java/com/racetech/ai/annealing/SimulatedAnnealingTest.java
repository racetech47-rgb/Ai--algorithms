package com.racetech.ai.annealing;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the simulated-annealing components.
 *
 * <p>All tests are self-contained and use simple mathematical objectives.
 */
class SimulatedAnnealingTest {

    // -----------------------------------------------------------------------
    // Cooling schedules
    // -----------------------------------------------------------------------

    /** Linear cooling must return exactly {@code initialTemp} at iteration 0. */
    @Test
    void testLinearCoolingAtIterationZero() {
        CoolingSchedule cs = new LinearCooling(1000);
        assertEquals(100.0, cs.computeTemperature(100.0, 0), 1e-9);
    }

    /** Exponential cooling must return exactly {@code initialTemp} at iteration 0. */
    @Test
    void testExponentialCoolingAtIterationZero() {
        CoolingSchedule cs = new ExponentialCooling(0.99);
        assertEquals(100.0, cs.computeTemperature(100.0, 0), 1e-9);
    }

    /** Logarithmic cooling must return exactly {@code initialTemp} at iteration 0. */
    @Test
    void testLogarithmicCoolingAtIterationZero() {
        CoolingSchedule cs = new LogarithmicCooling(1.0);
        assertEquals(100.0, cs.computeTemperature(100.0, 0), 1e-9);
    }

    /** All three cooling schedules must be strictly decreasing over iterations. */
    @Test
    void testCoolingSchedules() {
        CoolingSchedule[] schedules = {
            new LinearCooling(1000),
            new ExponentialCooling(0.95),
            new LogarithmicCooling(1.0),
        };

        for (CoolingSchedule cs : schedules) {
            double prev = cs.computeTemperature(100.0, 0);
            for (int iter = 1; iter <= 10; iter++) {
                double curr = cs.computeTemperature(100.0, iter);
                assertTrue(curr <= prev,
                    cs.getClass().getSimpleName() + " must produce non-increasing temperatures.");
                prev = curr;
            }
        }
    }

    /** Invalid constructor arguments must throw {@link IllegalArgumentException}. */
    @Test
    void testCoolingScheduleValidation() {
        assertThrows(IllegalArgumentException.class, () -> new LinearCooling(0));
        assertThrows(IllegalArgumentException.class, () -> new ExponentialCooling(1.0));
        assertThrows(IllegalArgumentException.class, () -> new ExponentialCooling(0.0));
        assertThrows(IllegalArgumentException.class, () -> new LogarithmicCooling(0.0));
    }

    // -----------------------------------------------------------------------
    // SimulatedAnnealing – acceptance probability
    // -----------------------------------------------------------------------

    /** A better (lower) cost must always be accepted (probability == 1). */
    @Test
    void testAcceptanceProbabilityBetterSolution() {
        double p = SimulatedAnnealing.acceptanceProbability(10.0, 8.0, 1.0);
        assertEquals(1.0, p, 1e-9,
            "A better solution must always be accepted.");
    }

    /** An equal cost must always be accepted (probability == 1). */
    @Test
    void testAcceptanceProbabilityEqualCost() {
        double p = SimulatedAnnealing.acceptanceProbability(5.0, 5.0, 2.0);
        assertEquals(1.0, p, 1e-9,
            "An equal-cost solution must always be accepted.");
    }

    /** A worse solution must have acceptance probability strictly in (0, 1). */
    @Test
    void testAcceptanceProbabilityWorseSolution() {
        double p = SimulatedAnnealing.acceptanceProbability(5.0, 10.0, 2.0);
        assertTrue(p > 0.0 && p < 1.0,
            "A worse solution must have probability in (0, 1).");
    }

    /** Higher temperature must increase the acceptance probability for a worse solution. */
    @Test
    void testAcceptanceProbability() {
        double pLow  = SimulatedAnnealing.acceptanceProbability(5.0, 10.0, 1.0);
        double pHigh = SimulatedAnnealing.acceptanceProbability(5.0, 10.0, 10.0);
        assertTrue(pHigh > pLow,
            "Higher temperature must increase acceptance probability.");
    }

    // -----------------------------------------------------------------------
    // SimulatedAnnealing – optimisation
    // -----------------------------------------------------------------------

    /**
     * Minimise a simple quadratic (x − 3)² represented as a single-element list.
     * The optimiser should converge close to x = 3.
     */
    @Test
    void testOptimizesSimpleFunction() {
        SimulatedAnnealing<List<Double>> sa = new SimulatedAnnealing<>(
            sol -> {
                double x = sol.get(0);
                return (x - 3.0) * (x - 3.0);
            },
            sol -> {
                double x = sol.get(0) + (Math.random() * 2.0 - 1.0) * 0.5;
                List<Double> next = new ArrayList<>();
                next.add(x);
                return next;
            },
            100.0,    // initialTemp
            0.001,    // minTemp
            5_000,    // maxIterations
            new ExponentialCooling(0.999)
        );

        List<Double> initial = new ArrayList<>();
        initial.add(0.0);

        AnnealingResult<List<Double>> result = sa.optimize(initial);

        assertEquals(0.0, result.getBestCost(), 0.5,
            "SA should find a solution near the minimum of (x-3)^2.");
    }

    /** The cost history must be non-empty and each entry must be non-negative. */
    @Test
    void testResultHasHistory() {
        SimulatedAnnealing<List<Double>> sa = new SimulatedAnnealing<>(
            sol -> sol.get(0) * sol.get(0),
            sol -> {
                List<Double> next = new ArrayList<>();
                next.add(sol.get(0) + (Math.random() - 0.5));
                return next;
            },
            10.0,
            0.01,
            200,
            new LinearCooling(200)
        );

        List<Double> initial = new ArrayList<>();
        initial.add(5.0);

        AnnealingResult<List<Double>> result = sa.optimize(initial);

        assertFalse(result.getCostHistory().isEmpty(),
            "Cost history must not be empty after optimisation.");
        for (double cost : result.getCostHistory()) {
            assertTrue(cost >= 0.0,
                "All recorded costs must be non-negative for x^2.");
        }
    }

    /** The best cost must not be worse than the cost of the initial solution. */
    @Test
    void testBestCostNotWorseThanInitial() {
        List<Double> initial = new ArrayList<>();
        initial.add(10.0);
        double initialCost = 10.0 * 10.0;   // x^2 at x=10

        SimulatedAnnealing<List<Double>> sa = new SimulatedAnnealing<>(
            sol -> sol.get(0) * sol.get(0),
            sol -> {
                List<Double> next = new ArrayList<>();
                next.add(sol.get(0) + (Math.random() - 0.5));
                return next;
            },
            50.0,
            0.01,
            1_000,
            new ExponentialCooling(0.99)
        );

        AnnealingResult<List<Double>> result = sa.optimize(initial);

        assertTrue(result.getBestCost() <= initialCost,
            "Best cost must not be worse than the initial solution's cost.");
    }
}
