package com.racetech.ai.annealing;

/**
 * Linear cooling schedule.
 *
 * <p>The temperature decreases linearly from the initial value to zero over
 * {@code totalIterations} steps:
 * <pre>
 *   T(k) = T₀ × (1 − k / totalIterations)
 * </pre>
 *
 * <p>Temperature is clamped to a small positive value ({@code 1e-10}) so it
 * never reaches exactly zero.
 */
public class LinearCooling implements CoolingSchedule {

    private final int totalIterations;

    /**
     * Creates a linear cooling schedule.
     *
     * @param totalIterations total number of iterations over which to cool;
     *                        must be &gt; 0
     * @throws IllegalArgumentException if {@code totalIterations} is not positive
     */
    public LinearCooling(int totalIterations) {
        if (totalIterations <= 0) {
            throw new IllegalArgumentException("totalIterations must be positive.");
        }
        this.totalIterations = totalIterations;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns {@code initialTemp * (1 - iteration / totalIterations)}, clamped
     * to a minimum of {@code 1e-10}.
     */
    @Override
    public double computeTemperature(double initialTemp, int iteration) {
        double t = initialTemp * (1.0 - (double) iteration / totalIterations);
        return Math.max(t, 1e-10);
    }
}
