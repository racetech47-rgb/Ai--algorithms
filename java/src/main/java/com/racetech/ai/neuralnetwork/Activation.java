package com.racetech.ai.neuralnetwork;

/**
 * Activation functions for neural-network layers.
 *
 * <p>Each constant exposes three operations:
 * <ul>
 *   <li>{@link #apply(double)} – element-wise activation</li>
 *   <li>{@link #applyDerivative(double)} – element-wise derivative (w.r.t. the pre-activation)</li>
 *   <li>{@link #applyVector(double[])} – vector activation (meaningful for SOFTMAX; for others
 *       it applies {@link #apply(double)} element-wise)</li>
 * </ul>
 */
public enum Activation {

    /** Rectified Linear Unit: max(0, x). */
    RELU {
        @Override
        public double apply(double x) {
            return Math.max(0.0, x);
        }

        @Override
        public double applyDerivative(double x) {
            return x > 0.0 ? 1.0 : 0.0;
        }
    },

    /** Logistic sigmoid: 1 / (1 + e^(-x)). */
    SIGMOID {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double applyDerivative(double x) {
            double s = apply(x);
            return s * (1.0 - s);
        }
    },

    /** Hyperbolic tangent. */
    TANH {
        @Override
        public double apply(double x) {
            return Math.tanh(x);
        }

        @Override
        public double applyDerivative(double x) {
            double t = Math.tanh(x);
            return 1.0 - t * t;
        }
    },

    /**
     * Softmax: converts a vector of logits to a probability distribution.
     *
     * <p>For SOFTMAX, use {@link #applyVector(double[])} rather than {@link #apply(double)}.
     * The scalar {@link #apply(double)} and {@link #applyDerivative(double)} are provided only
     * for interface completeness and treat the input as a single-element vector.
     */
    SOFTMAX {
        @Override
        public double apply(double x) {
            // Degenerate single-element case: softmax([x]) = 1.0.
            return 1.0;
        }

        @Override
        public double applyDerivative(double x) {
            double s = apply(x);
            return s * (1.0 - s);
        }

        @Override
        public double[] applyVector(double[] x) {
            double max = Double.NEGATIVE_INFINITY;
            for (double v : x) {
                if (v > max) max = v;
            }
            double sum = 0.0;
            double[] out = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                out[i] = Math.exp(x[i] - max);
                sum += out[i];
            }
            for (int i = 0; i < out.length; i++) {
                out[i] /= sum;
            }
            return out;
        }
    };

    /**
     * Applies the activation function element-wise to a scalar.
     *
     * @param x pre-activation value
     * @return activated value
     */
    public abstract double apply(double x);

    /**
     * Returns the derivative of the activation function evaluated at {@code x}.
     *
     * @param x pre-activation value
     * @return derivative at {@code x}
     */
    public abstract double applyDerivative(double x);

    /**
     * Applies the activation function to a vector.
     *
     * <p>For all activations except {@link #SOFTMAX} this is equivalent to calling
     * {@link #apply(double)} on each element.
     *
     * @param x input vector
     * @return activated vector (new array)
     */
    public double[] applyVector(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = apply(x[i]);
        }
        return out;
    }
}
