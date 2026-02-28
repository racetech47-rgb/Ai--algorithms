package com.racetech.ai.annealing;

/**
 * Strategy interface for computing the temperature at each iteration of
 * simulated annealing.
 *
 * <p>Implementations define how the temperature decreases (or "cools") over
 * the course of the optimisation run.  A higher temperature means the
 * algorithm is more likely to accept worse solutions; a lower temperature
 * means it is more conservative.
 */
public interface CoolingSchedule {

    /**
     * Computes the temperature for the given iteration.
     *
     * @param initialTemp the starting temperature (must be &gt; 0)
     * @param iteration   zero-based iteration index
     * @return current temperature (should be &gt; 0)
     */
    double computeTemperature(double initialTemp, int iteration);
}
