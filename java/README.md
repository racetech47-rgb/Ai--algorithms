# AI Algorithms Suite — Java

A Java 11 implementation of core AI algorithms: a feed-forward **Neural Network** with
backpropagation and a generic **Simulated Annealing** optimiser.

## Project layout

```
java/
├── pom.xml
└── src/
    ├── main/java/com/racetech/ai/
    │   ├── neuralnetwork/          # Neural-network core + Iris example
    │   └── annealing/             # Simulated-annealing core + TSP example
    └── test/java/com/racetech/ai/
        ├── neuralnetwork/         # JUnit 5 neural-network tests
        └── annealing/             # JUnit 5 annealing tests
```

## Build & test

```bash
cd java
mvn clean test          # compile and run all tests
mvn package             # produce ai-algorithms-1.0.0.jar
```

## Running the examples

```bash
# Iris classifier
mvn exec:java -Dexec.mainClass="com.racetech.ai.neuralnetwork.examples.IrisExample"

# Travelling-salesman problem
mvn exec:java -Dexec.mainClass="com.racetech.ai.annealing.examples.TspExample"
```

## Modules

### Neural Network (`com.racetech.ai.neuralnetwork`)

| Class | Purpose |
|---|---|
| `Activation` | RELU, SIGMOID, TANH, SOFTMAX activation functions |
| `Layer` | Single dense layer with He-initialised weights and backprop |
| `NeuralNetwork` | Multi-layer network: forward pass, MSE backprop, accuracy |
| `Trainer` | Mini-batch training loop with loss history |
| `IrisExample` | Demo: classifies a hard-coded Iris subset |

### Simulated Annealing (`com.racetech.ai.annealing`)

| Class / Interface | Purpose |
|---|---|
| `CoolingSchedule` | Strategy interface for temperature schedules |
| `LinearCooling` | T decreases linearly with iteration |
| `ExponentialCooling` | T decays exponentially |
| `LogarithmicCooling` | T decays logarithmically |
| `SimulatedAnnealing<S>` | Generic SA optimiser |
| `AnnealingResult<S>` | Holds best solution, cost, and full cost history |
| `TspExample` | Demo: solves a 5-city TSP |

## Requirements

- Java 11+
- Maven 3.6+
