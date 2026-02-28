"""TF-IDF + Logistic Regression text classifier."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


class TextClassifier:
    """Multi-class text classifier based on TF-IDF features and
    Logistic Regression.

    Args:
        num_classes: Number of distinct target classes. Used for
            validation and reporting only; inferred automatically
            during :meth:`fit`.

    Example:
        >>> clf = TextClassifier(num_classes=2)
        >>> clf.fit(["good product", "bad product"], [1, 0])
        >>> clf.predict(["great item"])
        [1]
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "TextClassifier":
        """Fit the TF-IDF vectoriser and Logistic Regression classifier.

        Args:
            texts: List of raw text samples.
            labels: Integer class labels corresponding to each sample.

        Returns:
            The fitted :class:`TextClassifier` instance.
        """
        self._pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, texts: List[str]) -> List[int]:
        """Predict class labels for a list of texts.

        Args:
            texts: List of raw text samples.

        Returns:
            List of integer class predictions.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._check_fitted()
        return self._pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return class probability estimates.

        Args:
            texts: List of raw text samples.

        Returns:
            Array of shape (n_samples, n_classes) with class
            probabilities.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._check_fitted()
        return self._pipeline.predict_proba(texts)

    def evaluate(
        self, texts: List[str], labels: List[int]
    ) -> Dict[str, float]:
        """Evaluate classifier performance on labelled data.

        Args:
            texts: List of raw text samples.
            labels: True integer class labels.

        Returns:
            Dictionary with keys ``'accuracy'``, ``'precision'``,
            ``'recall'``, and ``'f1'``.
        """
        self._check_fitted()
        preds = self.predict(texts)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before using the classifier.")
