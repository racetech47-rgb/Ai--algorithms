package com.racetech.ai.annealing;

/**
 * Logarithmic cooling schedule.
 *
 * <p>The temperature decreases logarithmically:
 * <pre>
 *   T(k) = T₀ / (1 + c × ln(1 + k))
 * </pre>
 * where {@code c} is a positive scaling constant.
 *
 * <p>Logarithmic cooling has theoretical guarantees of convergence to the
 * global optimum when {@code c} is sufficiently large, although it cools very
 * slowly in practice.
 */
public class LogarithmicCooling implements CoolingSchedule {

    private final double c;

    /**
     * Creates a logarithmic cooling schedule.
     *
     * @param c positive scaling constant; larger values produce slower cooling
     * @throws IllegalArgumentException if {@code c} is not positive
     */
    public LogarithmicCooling(double c) {
        if (c <= 0.0) {
            throw new IllegalArgumentException("Scaling constant c must be positive.");
        }
        this.c = c;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns {@code initialTemp / (1 + c * ln(1 + iteration))}.
     */
    @Override
    public double computeTemperature(double initialTemp, int iteration) {
        return initialTemp / (1.0 + c * Math.log(1.0 + iteration));
    }
}
