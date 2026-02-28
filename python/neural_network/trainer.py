"""Training utilities for MLP models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .mlp import MLP


class Trainer:
    """Handles training, evaluation, and inference for an MLP model.

    Args:
        model: The MLP instance to train.
        learning_rate: Learning rate for the optimizer.
        optimizer: Optimizer name ('adam' or 'sgd').

    Example:
        >>> model = MLP(4, [64, 32], 3)
        >>> trainer = Trainer(model, learning_rate=0.001)
    """

    def __init__(
        self,
        model: MLP,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
    ) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        optimizer = optimizer.lower()
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Use 'adam' or 'sgd'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensors(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        return X_t, y_t

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        X_t, y_t = self._to_tensors(X, y)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, List[float]]:
        """Train the model and return loss/accuracy history.

        Args:
            X_train: Training features of shape (n_samples, n_features).
            y_train: Integer class labels of shape (n_samples,).
            epochs: Number of full passes over the training data.
            batch_size: Mini-batch size.
            validation_data: Optional (X_val, y_val) tuple for validation.

        Returns:
            Dictionary with keys 'train_loss', 'train_acc', and optionally
            'val_loss', 'val_acc' containing per-epoch values.
        """
        loader = self._make_loader(X_train, y_train, batch_size)
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
        }
        if validation_data is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        for _ in range(epochs):
            self.model.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            history["train_loss"].append(epoch_loss / total)
            history["train_acc"].append(correct / total)

            if validation_data is not None:
                val_metrics = self.evaluate(validation_data[0], validation_data[1])
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])

        return history

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the model and return classification metrics.

        Args:
            X_test: Test features of shape (n_samples, n_features).
            y_test: True integer class labels of shape (n_samples,).

        Returns:
            Dictionary with keys: 'loss', 'accuracy', 'precision',
            'recall', 'f1'.
        """
        from sklearn.metrics import precision_recall_fscore_support

        self.model.eval()
        X_t, y_t = self._to_tensors(X_test, y_test)

        with torch.no_grad():
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t).item()
            preds = logits.argmax(dim=1).cpu().numpy()

        y_np = y_test if isinstance(y_test, np.ndarray) else y_t.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_np, preds, average="weighted", zero_division=0
        )
        accuracy = float((preds == y_np).mean())

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions for input data.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            Integer class predictions of shape (n_samples,).
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()
