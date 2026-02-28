"""Unit tests for the neural_network module."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from python.neural_network.mlp import MLP
from python.neural_network.trainer import Trainer


class TestMLP(unittest.TestCase):
    """Tests for the MLP model."""

    def _make_model(self, **kwargs) -> MLP:
        defaults = dict(input_size=8, hidden_sizes=[16, 8], output_size=3)
        defaults.update(kwargs)
        return MLP(**defaults)

    # -- Construction --

    def test_default_construction(self) -> None:
        model = self._make_model()
        self.assertIsInstance(model, MLP)

    def test_no_hidden_layers(self) -> None:
        model = MLP(input_size=4, hidden_sizes=[], output_size=2)
        self.assertIsInstance(model, MLP)

    def test_all_activations(self) -> None:
        for act in ("relu", "sigmoid", "tanh"):
            model = self._make_model(activation=act)
            self.assertIsInstance(model, MLP)

    def test_invalid_activation_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_model(activation="swish")

    def test_invalid_dropout_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_model(dropout=1.5)

    # -- Forward pass --

    def test_forward_output_shape(self) -> None:
        model = self._make_model(input_size=8, hidden_sizes=[16], output_size=3)
        x = torch.randn(10, 8)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([10, 3]))

    def test_forward_batch_size_one(self) -> None:
        model = self._make_model(input_size=8, hidden_sizes=[16], output_size=3)
        x = torch.randn(1, 8)
        out = model(x)
        self.assertEqual(out.shape, torch.Size([1, 3]))

    # -- Save / Load --

    def test_save_and_load(self) -> None:
        model = self._make_model()
        x = torch.randn(4, 8)
        original_out = model(x).detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = MLP.load(
                path,
                input_size=8,
                hidden_sizes=[16, 8],
                output_size=3,
            )
            loaded_out = loaded(x)
            self.assertTrue(torch.allclose(original_out, loaded_out, atol=1e-6))
        finally:
            os.unlink(path)


class TestTrainer(unittest.TestCase):
    """Tests for the Trainer class."""

    def _make_data(
        self,
        n: int = 60,
        features: int = 8,
        classes: int = 3,
    ):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, features)).astype(np.float32)
        y = rng.integers(0, classes, n)
        return X, y

    def _make_trainer(self, optimizer: str = "adam") -> Trainer:
        model = MLP(input_size=8, hidden_sizes=[16, 8], output_size=3)
        return Trainer(model, learning_rate=0.01, optimizer=optimizer)

    # -- Optimizer creation --

    def test_adam_optimizer(self) -> None:
        trainer = self._make_trainer("adam")
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)

    def test_sgd_optimizer(self) -> None:
        trainer = self._make_trainer("sgd")
        self.assertIsInstance(trainer.optimizer, torch.optim.SGD)

    def test_invalid_optimizer_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_trainer("rmsprop")

    # -- Training --

    def test_train_returns_history_keys(self) -> None:
        trainer = self._make_trainer()
        X, y = self._make_data()
        history = trainer.train(X, y, epochs=3, batch_size=16)
        for key in ("train_loss", "train_acc"):
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), 3)

    def test_train_with_validation(self) -> None:
        trainer = self._make_trainer()
        X, y = self._make_data(60)
        history = trainer.train(
            X[:40], y[:40], epochs=2, batch_size=16,
            validation_data=(X[40:], y[40:])
        )
        for key in ("val_loss", "val_acc"):
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), 2)

    # -- Evaluation --

    def test_evaluate_returns_expected_keys(self) -> None:
        trainer = self._make_trainer()
        X, y = self._make_data()
        trainer.train(X, y, epochs=1, batch_size=16)
        metrics = trainer.evaluate(X, y)
        for key in ("loss", "accuracy", "precision", "recall", "f1"):
            self.assertIn(key, metrics)

    def test_evaluate_accuracy_in_range(self) -> None:
        trainer = self._make_trainer()
        X, y = self._make_data()
        trainer.train(X, y, epochs=5, batch_size=16)
        metrics = trainer.evaluate(X, y)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    # -- Predict --

    def test_predict_shape(self) -> None:
        trainer = self._make_trainer()
        X, y = self._make_data(20)
        trainer.train(X, y, epochs=1, batch_size=16)
        preds = trainer.predict(X)
        self.assertEqual(len(preds), 20)


if __name__ == "__main__":
    unittest.main()
