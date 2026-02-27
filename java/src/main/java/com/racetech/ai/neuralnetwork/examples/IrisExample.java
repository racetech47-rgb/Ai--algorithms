package com.racetech.ai.neuralnetwork.examples;

import com.racetech.ai.neuralnetwork.Activation;
import com.racetech.ai.neuralnetwork.NeuralNetwork;
import com.racetech.ai.neuralnetwork.Trainer;

import java.util.List;
import java.util.Map;

/**
 * Demonstrates the neural network on a small, hard-coded subset of the Iris dataset.
 *
 * <p>Ten samples (4 features, 3 classes) are used so that the example runs
 * instantly without any external data files.
 */
public class IrisExample {

    public static void main(String[] args) {
        // --- 10-sample Iris subset (sepal-length, sepal-width, petal-length, petal-width) ---
        // Normalised to [0,1] using approximate min/max from the full dataset.
        double[][] X = {
            {0.222, 0.625, 0.068, 0.042},  // class 0 – Iris-setosa
            {0.167, 0.417, 0.068, 0.042},
            {0.111, 0.500, 0.051, 0.042},
            {0.083, 0.458, 0.085, 0.042},
            {0.194, 0.667, 0.068, 0.042},
            {0.306, 0.583, 0.441, 0.542},  // class 1 – Iris-versicolor
            {0.222, 0.542, 0.441, 0.583},
            {0.750, 0.500, 0.627, 0.542},  // class 2 – Iris-virginica
            {0.694, 0.417, 0.593, 0.583},
            {0.667, 0.458, 0.627, 0.708},
        };

        // One-hot encoded targets.
        double[][] yOneHot = {
            {1, 0, 0},
            {1, 0, 0},
            {1, 0, 0},
            {1, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 0, 1},
            {0, 0, 1},
        };

        int[] yTrue = {0, 0, 0, 0, 0, 1, 1, 2, 2, 2};

        // Build network: 4 → 8 → 3 with RELU hidden, SOFTMAX output.
        NeuralNetwork network = new NeuralNetwork(
            new int[]        {4, 8, 3},
            new Activation[] {Activation.RELU, Activation.SOFTMAX},
            0.01
        );

        Trainer trainer = new Trainer(network);

        System.out.println("=== Iris Example ===");
        System.out.println("Training for 100 epochs on 10 samples...");

        Map<String, List<Double>> history = trainer.train(X, yOneHot, 100, 5);

        List<Double> losses = history.get("loss");
        System.out.printf("Initial loss : %.6f%n", losses.get(0));
        System.out.printf("Final   loss : %.6f%n", losses.get(losses.size() - 1));

        Map<String, Double> metrics = trainer.evaluate(X, yTrue);
        System.out.printf("Accuracy     : %.1f%%%n", metrics.get("accuracy") * 100);

        System.out.println("\nPer-sample predictions:");
        String[] classNames = {"setosa", "versicolor", "virginica"};
        for (int i = 0; i < X.length; i++) {
            int pred = network.predict(X[i]);
            System.out.printf("  Sample %2d → predicted: %-12s  actual: %s%n",
                i, classNames[pred], classNames[yTrue[i]]);
        }
    }
}
