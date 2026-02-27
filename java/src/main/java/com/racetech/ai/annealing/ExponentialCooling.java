package com.racetech.ai.annealing;

/**
 * Exponential cooling schedule.
 *
 * <p>The temperature decays exponentially with each iteration:
 * <pre>
 *   T(k) = T₀ × α^k
 * </pre>
 * where {@code α} (alpha) is the decay rate in the range {@code (0, 1)}.
 *
 * <p>This is the most commonly used cooling schedule in practice because it
 * provides smooth, controlled convergence.
 */
public class ExponentialCooling implements CoolingSchedule {

    private final double alpha;

    /**
     * Creates an exponential cooling schedule.
     *
     * @param alpha decay rate; must be in the open interval {@code (0, 1)}
     * @throws IllegalArgumentException if {@code alpha} is not in {@code (0, 1)}
     */
    public ExponentialCooling(double alpha) {
        if (alpha <= 0.0 || alpha >= 1.0) {
            throw new IllegalArgumentException("alpha must be in the open interval (0, 1).");
        }
        this.alpha = alpha;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns {@code initialTemp * alpha^iteration}.
     */
    @Override
    public double computeTemperature(double initialTemp, int iteration) {
        return initialTemp * Math.pow(alpha, iteration);
    }
}
