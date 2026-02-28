package com.racetech.ai.neuralnetwork;

import java.util.Random;

/**
 * A single fully-connected (dense) layer in a neural network.
 *
 * <p>Weights are He-initialised; biases start at zero.  The layer caches the
 * last input, pre-activation, and output so that {@link #backward} can
 * compute gradients without needing them passed in again.
 */
public class Layer {

    private final double[][] weights;       // [outputSize][inputSize]
    private final double[]   biases;        // [outputSize]
    private final Activation activation;
    private final int        inputSize;
    private final int        outputSize;

    // --- cached values for backpropagation ---
    private double[] lastInput;
    private double[] lastPreactivation;
    private double[] lastOutput;

    /**
     * Constructs a layer and He-initialises its weights.
     *
     * @param inputSize  number of input neurons
     * @param outputSize number of output neurons
     * @param activation activation function to apply after the linear transform
     */
    public Layer(int inputSize, int outputSize, Activation activation) {
        this.inputSize  = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        this.weights    = new double[outputSize][inputSize];
        this.biases     = new double[outputSize];
        heInitialise();
    }

    /** He (Kaiming) weight initialisation: W ~ N(0, sqrt(2 / fanIn)). */
    private void heInitialise() {
        Random rng   = new Random(42);
        double stdDev = Math.sqrt(2.0 / inputSize);
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[o][i] = rng.nextGaussian() * stdDev;
            }
        }
    }

    /**
     * Performs the forward pass through this layer.
     *
     * <p>Computes {@code output = activation(W · input + b)} and caches
     * the intermediate values for backpropagation.
     *
     * @param input activation vector from the previous layer (length {@code inputSize})
     * @return output activation vector (length {@code outputSize})
     */
    public double[] forward(double[] input) {
        lastInput = input.clone();

        double[] pre = new double[outputSize];
        for (int o = 0; o < outputSize; o++) {
            pre[o] = biases[o];
            for (int i = 0; i < inputSize; i++) {
                pre[o] += weights[o][i] * input[i];
            }
        }
        lastPreactivation = pre;
        lastOutput = activation.applyVector(pre);
        return lastOutput.clone();
    }

    /**
     * Performs the backward pass (backpropagation) through this layer.
     *
     * <p>Given the gradient of the loss w.r.t. this layer's output
     * ({@code gradOutput}), this method:
     * <ol>
     *   <li>Multiplies element-wise by the activation derivative to obtain the
     *       gradient w.r.t. the pre-activation (delta).</li>
     *   <li>Updates weights and biases by gradient descent.</li>
     *   <li>Returns the gradient w.r.t. this layer's input so it can be
     *       propagated to the previous layer.</li>
     * </ol>
     *
     * @param gradOutput   gradient of the loss w.r.t. this layer's output
     * @param learningRate step size for the weight update
     * @return gradient of the loss w.r.t. this layer's input (length {@code inputSize})
     */
    public double[] backward(double[] gradOutput, double learningRate) {
        // Delta: grad w.r.t. pre-activation.
        double[] delta = new double[outputSize];
        for (int o = 0; o < outputSize; o++) {
            delta[o] = gradOutput[o] * activation.applyDerivative(lastPreactivation[o]);
        }

        // Gradient w.r.t. input to propagate further back.
        double[] gradInput = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int o = 0; o < outputSize; o++) {
                gradInput[i] += weights[o][i] * delta[o];
            }
        }

        // Update weights and biases.
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[o][i] -= learningRate * delta[o] * lastInput[i];
            }
            biases[o] -= learningRate * delta[o];
        }

        return gradInput;
    }

    /**
     * Returns the most-recently computed output vector (after {@link #forward}).
     *
     * @return last output activation (length {@code outputSize})
     */
    public double[] getLastOutput() {
        return lastOutput != null ? lastOutput.clone() : new double[outputSize];
    }

    /** @return number of input neurons */
    public int getInputSize()  { return inputSize; }

    /** @return number of output neurons */
    public int getOutputSize() { return outputSize; }

    /** @return the activation function used by this layer */
    public Activation getActivation() { return activation; }
}
