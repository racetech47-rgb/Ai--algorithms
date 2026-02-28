package com.racetech.ai.annealing;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * Generic simulated-annealing optimiser.
 *
 * <p>Simulated annealing is a probabilistic technique for approximating the
 * global minimum of a function.  Starting from an initial solution it
 * iteratively proposes a neighbouring solution and accepts it according to
 * the Metropolis criterion:
 * <ul>
 *   <li>If the new solution is better (lower cost) it is always accepted.</li>
 *   <li>If it is worse it is accepted with probability
 *       {@code exp(-(newCost - currentCost) / T)}, where {@code T} is the
 *       current temperature.</li>
 * </ul>
 *
 * @param <S> type of the solution (e.g. {@code List<Integer>} for a TSP tour)
 */
public class SimulatedAnnealing<S> {

    private final Function<S, Double> objective;
    private final Function<S, S>      neighbor;
    private final double              initialTemp;
    private final double              minTemp;
    private final int                 maxIterations;
    private final CoolingSchedule     cooling;
    private final Random              rng;

    /**
     * Creates a new simulated-annealing optimiser.
     *
     * @param objective     function that computes the cost of a solution (lower is better)
     * @param neighbor      function that generates a neighbouring solution from a current one
     * @param initialTemp   starting temperature (must be &gt; 0)
     * @param minTemp       minimum temperature at which the algorithm stops (must be &gt; 0)
     * @param maxIterations hard upper bound on the number of iterations
     * @param cooling       cooling schedule that maps (initialTemp, iteration) → temperature
     */
    public SimulatedAnnealing(
            Function<S, Double> objective,
            Function<S, S>      neighbor,
            double              initialTemp,
            double              minTemp,
            int                 maxIterations,
            CoolingSchedule     cooling) {

        this.objective     = objective;
        this.neighbor      = neighbor;
        this.initialTemp   = initialTemp;
        this.minTemp       = minTemp;
        this.maxIterations = maxIterations;
        this.cooling       = cooling;
        this.rng           = new Random(42);
    }

    /**
     * Runs the optimisation from the given initial solution.
     *
     * @param initialSolution starting point for the search
     * @return an {@link AnnealingResult} containing the best solution found,
     *         its cost, and the full per-iteration cost history
     */
    public AnnealingResult<S> optimize(S initialSolution) {
        S      current     = initialSolution;
        double currentCost = objective.apply(current);

        S      best     = current;
        double bestCost = currentCost;

        List<Double> history = new ArrayList<>(maxIterations);

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double temp = cooling.computeTemperature(initialTemp, iteration);
            if (temp < minTemp) break;

            S      candidate     = neighbor.apply(current);
            double candidateCost = objective.apply(candidate);

            if (acceptanceProbability(currentCost, candidateCost, temp) > rng.nextDouble()) {
                current     = candidate;
                currentCost = candidateCost;
            }

            if (currentCost < bestCost) {
                best     = current;
                bestCost = currentCost;
            }

            history.add(bestCost);
        }

        return new AnnealingResult<>(best, bestCost, history);
    }

    /**
     * Computes the Metropolis acceptance probability.
     *
     * <p>Returns {@code 1.0} if the new cost is lower than or equal to the
     * current cost; otherwise returns {@code exp(-(newCost - currentCost) / temp)}.
     *
     * @param currentCost cost of the current solution
     * @param newCost     cost of the proposed neighbouring solution
     * @param temp        current temperature (must be &gt; 0)
     * @return acceptance probability in the range {@code (0, 1]}
     */
    public static double acceptanceProbability(
            double currentCost, double newCost, double temp) {
        if (newCost <= currentCost) return 1.0;
        return Math.exp(-(newCost - currentCost) / temp);
    }
}
