"""Iris dataset classification example using MLP."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.neural_network import MLP, Trainer


def main() -> None:
    """Load Iris data, train an MLP, and print evaluation results."""
    # ── Data preparation ────────────────────────────────────────────────
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # ── Model & trainer ─────────────────────────────────────────────────
    model = MLP(
        input_size=4,
        hidden_sizes=[64, 32],
        output_size=3,
        activation="relu",
        dropout=0.2,
    )
    trainer = Trainer(model, learning_rate=0.001, optimizer="adam")

    # ── Training ────────────────────────────────────────────────────────
    print("Training MLP on Iris dataset …")
    history = trainer.train(
        X_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
    )
    print(
        f"  Final train acc : {history['train_acc'][-1]:.4f} | "
        f"val acc : {history['val_acc'][-1]:.4f}"
    )

    # ── Evaluation ──────────────────────────────────────────────────────
    metrics = trainer.evaluate(X_test, y_test)
    print("\nTest-set metrics:")
    for key, value in metrics.items():
        print(f"  {key:<12}: {value:.4f}")


if __name__ == "__main__":
    main()
