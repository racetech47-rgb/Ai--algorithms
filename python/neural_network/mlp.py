"""Multi-Layer Perceptron (MLP) implementation using PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """A configurable Multi-Layer Perceptron for classification tasks.

    Args:
        input_size: Number of input features.
        hidden_sizes: List of hidden layer sizes.
        output_size: Number of output classes.
        activation: Activation function name ('relu', 'sigmoid', 'tanh').
        dropout: Dropout probability applied after each hidden layer.

    Example:
        >>> model = MLP(input_size=4, hidden_sizes=[64, 32], output_size=3)
        >>> x = torch.randn(16, 4)
        >>> out = model(x)
        >>> out.shape
        torch.Size([16, 3])
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS.keys())}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        layers: List[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._ACTIVATIONS[activation]())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Raw logits tensor of shape (batch_size, output_size).
        """
        return self.network(x)

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: File path for saving the state dict (e.g. 'model.pt').
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> "MLP":
        """Load a previously saved MLP from disk.

        Args:
            path: File path of the saved state dict.
            input_size: Number of input features (must match saved model).
            hidden_sizes: Hidden layer sizes (must match saved model).
            output_size: Number of output classes (must match saved model).
            activation: Activation function used in the saved model.
            dropout: Dropout probability used in the saved model.

        Returns:
            MLP instance with loaded weights.
        """
        model = cls(input_size, hidden_sizes, output_size, activation, dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
