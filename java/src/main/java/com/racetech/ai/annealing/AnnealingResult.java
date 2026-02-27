package com.racetech.ai.annealing;

import java.util.Collections;
import java.util.List;

/**
 * Immutable container for the result of a simulated-annealing run.
 *
 * @param <S> type of the solution
 */
public class AnnealingResult<S> {

    private final S             bestSolution;
    private final double        bestCost;
    private final List<Double>  costHistory;

    /**
     * Constructs an annealing result.
     *
     * @param bestSolution the best solution found during the run
     * @param bestCost     the objective value of the best solution
     * @param costHistory  per-iteration best-cost values recorded during the run
     */
    public AnnealingResult(S bestSolution, double bestCost, List<Double> costHistory) {
        this.bestSolution = bestSolution;
        this.bestCost     = bestCost;
        this.costHistory  = Collections.unmodifiableList(costHistory);
    }

    /**
     * Returns the best solution found.
     *
     * @return best solution
     */
    public S getBestSolution() {
        return bestSolution;
    }

    /**
     * Returns the objective (cost) value of the best solution.
     *
     * @return best cost
     */
    public double getBestCost() {
        return bestCost;
    }

    /**
     * Returns an unmodifiable list of the best-cost value recorded at each
     * iteration of the annealing run.
     *
     * @return cost history list
     */
    public List<Double> getCostHistory() {
        return costHistory;
    }

    @Override
    public String toString() {
        return String.format("AnnealingResult{bestCost=%.6f, iterations=%d}",
            bestCost, costHistory.size());
    }
}
