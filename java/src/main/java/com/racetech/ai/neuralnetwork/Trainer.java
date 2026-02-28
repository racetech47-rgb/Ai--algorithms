package com.racetech.ai.neuralnetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Trains a {@link NeuralNetwork} using mini-batch gradient descent.
 *
 * <p>Loss history is recorded per epoch and returned after training so callers
 * can inspect convergence.
 */
public class Trainer {

    private final NeuralNetwork network;
    private final Random        rng;

    /**
     * Creates a trainer bound to the given network.
     *
     * @param network the network to train
     */
    public Trainer(NeuralNetwork network) {
        this.network = network;
        this.rng     = new Random(0);
    }

    /**
     * Trains the network for the specified number of epochs.
     *
     * <p>In each epoch the training set is shuffled and processed in
     * mini-batches of size {@code batchSize}.  The average MSE loss per epoch
     * is stored in the returned history map under the key {@code "loss"}.
     *
     * @param X         feature matrix ({@code [numSamples][numFeatures]})
     * @param y         one-hot encoded target matrix ({@code [numSamples][numClasses]})
     * @param epochs    number of full passes over the training set
     * @param batchSize number of samples per mini-batch (use 1 for pure SGD)
     * @return map containing {@code "loss"} → list of per-epoch average losses
     */
    public Map<String, List<Double>> train(
            double[][] X, double[][] y, int epochs, int batchSize) {

        int numSamples = X.length;
        List<Integer> indices = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) indices.add(i);

        List<Double> lossHistory = new ArrayList<>(epochs);

        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(indices, rng);

            double epochLoss = 0.0;
            int    batches   = 0;

            for (int start = 0; start < numSamples; start += batchSize) {
                int end        = Math.min(start + batchSize, numSamples);
                double batchLoss = 0.0;

                for (int idx = start; idx < end; idx++) {
                    int s = indices.get(idx);
                    network.forward(X[s]);
                    batchLoss += network.computeLoss(y[s]);
                    network.backward(y[s]);
                }

                epochLoss += batchLoss / (end - start);
                batches++;
            }

            lossHistory.add(epochLoss / batches);
        }

        Map<String, List<Double>> history = new HashMap<>();
        history.put("loss", Collections.unmodifiableList(lossHistory));
        return Collections.unmodifiableMap(history);
    }

    /**
     * Evaluates classification accuracy on the given dataset.
     *
     * @param X     feature matrix ({@code [numSamples][numFeatures]})
     * @param yTrue ground-truth class indices (length {@code numSamples})
     * @return map containing {@code "accuracy"} → fraction of correct predictions
     */
    public Map<String, Double> evaluate(double[][] X, int[] yTrue) {
        double accuracy = network.evaluateAccuracy(X, yTrue);
        Map<String, Double> result = new HashMap<>();
        result.put("accuracy", accuracy);
        return Collections.unmodifiableMap(result);
    }
}
