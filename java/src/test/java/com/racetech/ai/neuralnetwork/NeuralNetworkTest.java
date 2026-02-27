package com.racetech.ai.neuralnetwork;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the neural-network components.
 *
 * <p>All tests are self-contained and require no external data files.
 */
class NeuralNetworkTest {

    // -----------------------------------------------------------------------
    // Layer tests
    // -----------------------------------------------------------------------

    /** The output of {@link Layer#forward} must have exactly {@code outputSize} elements. */
    @Test
    void testLayerForwardShape() {
        Layer layer = new Layer(4, 8, Activation.RELU);
        double[] input  = {0.1, 0.2, 0.3, 0.4};
        double[] output = layer.forward(input);

        assertEquals(8, output.length,
            "forward() output length must equal the layer's outputSize.");
    }

    /** Every output of a RELU layer must be non-negative. */
    @Test
    void testReluOutputNonNegative() {
        Layer layer = new Layer(3, 5, Activation.RELU);
        double[] input = {-1.0, 0.5, 2.0};
        double[] output = layer.forward(input);
        for (double v : output) {
            assertTrue(v >= 0.0, "RELU output must be non-negative.");
        }
    }

    // -----------------------------------------------------------------------
    // NeuralNetwork – forward pass
    // -----------------------------------------------------------------------

    /** The forward-pass output vector must have the correct length. */
    @Test
    void testNeuralNetworkForward() {
        NeuralNetwork nn = new NeuralNetwork(
            new int[]        {4, 8, 3},
            new Activation[] {Activation.RELU, Activation.SOFTMAX},
            0.01
        );

        double[] input  = {0.5, 0.3, 0.8, 0.1};
        double[] output = nn.forward(input);

        assertEquals(3, output.length,
            "Network output length must equal the number of output neurons.");
    }

    /** SOFTMAX output values must sum to approximately 1.0. */
    @Test
    void testSoftmaxOutputSumsToOne() {
        NeuralNetwork nn = new NeuralNetwork(
            new int[]        {2, 4, 3},
            new Activation[] {Activation.RELU, Activation.SOFTMAX},
            0.01
        );

        double[] output = nn.forward(new double[]{0.6, 0.4});
        double sum = 0.0;
        for (double v : output) sum += v;
        assertEquals(1.0, sum, 1e-9, "SOFTMAX output must sum to 1.");
    }

    // -----------------------------------------------------------------------
    // NeuralNetwork – training
    // -----------------------------------------------------------------------

    /** Training for a few epochs must not throw any exception. */
    @Test
    void testNeuralNetworkTrainsWithoutError() {
        NeuralNetwork nn = new NeuralNetwork(
            new int[]        {2, 4, 2},
            new Activation[] {Activation.RELU, Activation.SIGMOID},
            0.01
        );

        double[][] X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        double[][] y = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};

        assertDoesNotThrow(() -> {
            for (int e = 0; e < 10; e++) {
                for (int i = 0; i < X.length; i++) {
                    nn.forward(X[i]);
                    nn.backward(y[i]);
                }
            }
        });
    }

    /** MSE loss should decrease (or stay the same) after training on simple data. */
    @Test
    void testLossDecreasesAfterTraining() {
        NeuralNetwork nn = new NeuralNetwork(
            new int[]        {2, 4, 1},
            new Activation[] {Activation.RELU, Activation.SIGMOID},
            0.05
        );

        double[][] X = {{0.2, 0.8}, {0.9, 0.1}};
        double[][] y = {{1.0}, {0.0}};

        // Measure initial loss.
        double initialLoss = 0.0;
        for (int i = 0; i < X.length; i++) {
            nn.forward(X[i]);
            initialLoss += nn.computeLoss(y[i]);
        }

        // Train for 200 epochs.
        for (int epoch = 0; epoch < 200; epoch++) {
            for (int i = 0; i < X.length; i++) {
                nn.forward(X[i]);
                nn.backward(y[i]);
            }
        }

        double finalLoss = 0.0;
        for (int i = 0; i < X.length; i++) {
            nn.forward(X[i]);
            finalLoss += nn.computeLoss(y[i]);
        }

        assertTrue(finalLoss <= initialLoss,
            "Loss should not increase after training.");
    }

    // -----------------------------------------------------------------------
    // NeuralNetwork – prediction
    // -----------------------------------------------------------------------

    /** {@link NeuralNetwork#predict} must return a valid class index. */
    @Test
    void testPredictReturnsValidClass() {
        int outputSize = 5;
        NeuralNetwork nn = new NeuralNetwork(
            new int[]        {3, 6, outputSize},
            new Activation[] {Activation.RELU, Activation.SOFTMAX},
            0.01
        );

        int predicted = nn.predict(new double[]{0.1, 0.5, 0.9});
        assertTrue(predicted >= 0 && predicted < outputSize,
            "predict() must return an index in [0, outputSize).");
    }

    /** {@link NeuralNetwork#evaluateAccuracy} must return a value in [0, 1]. */
    @Test
    void testEvaluateAccuracyInRange() {
        NeuralNetwork nn = new NeuralNetwork(
            new int[]        {2, 4, 3},
            new Activation[] {Activation.SIGMOID, Activation.SOFTMAX},
            0.01
        );

        double[][] X     = {{0.1, 0.2}, {0.9, 0.8}, {0.5, 0.5}};
        int[]      yTrue = {0, 1, 2};

        double accuracy = nn.evaluateAccuracy(X, yTrue);
        assertTrue(accuracy >= 0.0 && accuracy <= 1.0,
            "evaluateAccuracy() must return a value in [0, 1].");
    }

    // -----------------------------------------------------------------------
    // Constructor validation
    // -----------------------------------------------------------------------

    /** Passing inconsistent array lengths to the constructor must throw. */
    @Test
    void testConstructorValidatesArrayLengths() {
        assertThrows(IllegalArgumentException.class, () ->
            new NeuralNetwork(
                new int[]        {4, 8, 3},
                new Activation[] {Activation.RELU},   // should have length 2
                0.01
            )
        );
    }
}
