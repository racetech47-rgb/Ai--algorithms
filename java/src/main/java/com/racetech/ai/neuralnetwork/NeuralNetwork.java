package com.racetech.ai.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * A feed-forward neural network composed of fully-connected {@link Layer}s.
 *
 * <p>Training uses stochastic / mini-batch gradient descent with MSE loss.
 * The network supports multi-class classification via argmax of the output layer.
 */
public class NeuralNetwork {

    private final List<Layer> layers;
    private final double      learningRate;

    // Cache the last forward-pass output for backpropagation.
    private double[] lastOutput;

    /**
     * Constructs a neural network.
     *
     * @param layerSizes   array of sizes: {@code [inputSize, hidden1, ..., outputSize]}.
     *                     Must have at least two elements.
     * @param activations  activation function for each layer (length {@code layerSizes.length - 1})
     * @param learningRate gradient-descent step size
     * @throws IllegalArgumentException if array lengths are inconsistent
     */
    public NeuralNetwork(int[] layerSizes, Activation[] activations, double learningRate) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("layerSizes must contain at least two elements.");
        }
        if (activations.length != layerSizes.length - 1) {
            throw new IllegalArgumentException(
                "activations.length must equal layerSizes.length - 1.");
        }
        this.learningRate = learningRate;
        this.layers       = new ArrayList<>();
        for (int i = 0; i < activations.length; i++) {
            layers.add(new Layer(layerSizes[i], layerSizes[i + 1], activations[i]));
        }
    }

    /**
     * Runs the forward pass through all layers.
     *
     * @param input input feature vector (length must match the first layer's input size)
     * @return output activation vector from the final layer
     */
    public double[] forward(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        lastOutput = current;
        return current.clone();
    }

    /**
     * Runs backpropagation using MSE loss and updates all layer weights.
     *
     * <p>{@link #forward} must be called before {@code backward} so that the
     * layers' cached values are populated.
     *
     * @param target desired output vector (same length as the network's output layer)
     */
    public void backward(double[] target) {
        // MSE gradient: dL/dOutput = 2 * (output - target) / n
        double[] gradOutput = new double[lastOutput.length];
        for (int i = 0; i < lastOutput.length; i++) {
            gradOutput[i] = 2.0 * (lastOutput[i] - target[i]) / lastOutput.length;
        }

        // Propagate backwards through layers in reverse order.
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradOutput = layers.get(i).backward(gradOutput, learningRate);
        }
    }

    /**
     * Runs the forward pass and returns the index of the highest-valued output neuron.
     *
     * @param input input feature vector
     * @return predicted class index (argmax of output)
     */
    public int predict(double[] input) {
        double[] output = forward(input);
        int    bestIdx  = 0;
        double bestVal  = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > bestVal) {
                bestVal = output[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Computes classification accuracy over a dataset.
     *
     * @param X      feature matrix ({@code [numSamples][numFeatures]})
     * @param yTrue  ground-truth class indices (length {@code numSamples})
     * @return fraction of correctly classified samples in [0, 1]
     */
    public double evaluateAccuracy(double[][] X, int[] yTrue) {
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            if (predict(X[i]) == yTrue[i]) {
                correct++;
            }
        }
        return (double) correct / X.length;
    }

    /**
     * Computes MSE loss for a single sample (after calling {@link #forward}).
     *
     * @param target desired output vector
     * @return mean squared error
     */
    public double computeLoss(double[] target) {
        double loss = 0.0;
        for (int i = 0; i < lastOutput.length; i++) {
            double diff = lastOutput[i] - target[i];
            loss += diff * diff;
        }
        return loss / lastOutput.length;
    }

    /** @return unmodifiable view of the layers */
    public List<Layer> getLayers() {
        return List.copyOf(layers);
    }

    /** @return the learning rate used by this network */
    public double getLearningRate() {
        return learningRate;
    }
}
