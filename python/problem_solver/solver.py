"""Integrated problem solver that routes tasks to specialist sub-modules."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ProblemSolver:
    """High-level facade that routes problems to the correct algorithm.

    Supported problem types (for :meth:`auto_solve`):
    - ``"classification"`` — text classification via NLP pipeline.
    - ``"optimization"``   — TSP-style optimization via simulated annealing.
    - ``"sentiment"``      — sentiment analysis via transformer/lexicon.

    Example:
        >>> ps = ProblemSolver()
        >>> result = ps.auto_solve("sentiment", texts=["I love this!"])
    """

    def __init__(self) -> None:
        # Sub-components are lazily imported to avoid hard import errors
        # when optional dependencies are missing.
        pass

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def solve_classification(
        self,
        texts: List[str],
        labels: List[int],
        test_texts: List[str],
    ) -> Dict[str, Any]:
        """Train a text classifier and predict labels for test texts.

        Applies NLP preprocessing followed by a TF-IDF + Logistic
        Regression model.

        Args:
            texts: Training text samples.
            labels: Integer class labels for training samples.
            test_texts: Text samples to classify.

        Returns:
            Dictionary with keys:

            - ``"predictions"`` — list of predicted integer labels.
            - ``"probabilities"`` — probability array (n_test, n_classes).
            - ``"metrics"`` — evaluation metrics on the training set.
        """
        from python.nlp.preprocessing import TextPreprocessor
        from python.nlp.text_classification import TextClassifier

        num_classes = len(set(labels))
        preprocessor = TextPreprocessor()
        processed_train = preprocessor.batch_preprocess(texts)
        processed_test = preprocessor.batch_preprocess(test_texts)

        clf = TextClassifier(num_classes=num_classes)
        clf.fit(processed_train, labels)

        predictions = clf.predict(processed_test)
        probabilities = clf.predict_proba(processed_test)
        metrics = clf.evaluate(processed_train, labels)

        return {
            "predictions": predictions,
            "probabilities": probabilities.tolist(),
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Optimization (TSP)
    # ------------------------------------------------------------------

    def solve_optimization(
        self,
        cities: Optional[List[Tuple[float, float]]] = None,
        num_cities: int = 15,
    ) -> Dict[str, Any]:
        """Solve a TSP instance using simulated annealing.

        Args:
            cities: List of (x, y) city coordinates. If *None*, random
                cities are generated.
            num_cities: Number of random cities to generate when *cities*
                is not provided.

        Returns:
            Dictionary with keys:

            - ``"best_tour"`` — optimized city order (list of indices).
            - ``"best_length"`` — total tour length.
            - ``"initial_length"`` — tour length before optimization.
            - ``"improvement_pct"`` — percentage improvement.
            - ``"iterations"`` — number of SA iterations performed.
        """
        from python.simulated_annealing import SimulatedAnnealing, ExponentialCooling

        if cities is None:
            random.seed(0)
            cities = [
                (random.uniform(0, 100), random.uniform(0, 100))
                for _ in range(num_cities)
            ]

        def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def tour_length(tour: List[int]) -> float:
            return sum(
                _dist(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
                for i in range(len(tour))
            )

        def two_opt(tour: List[int]) -> List[int]:
            i, j = sorted(random.sample(range(len(tour)), 2))
            return tour[:i] + list(reversed(tour[i : j + 1])) + tour[j + 1 :]

        initial_tour = list(range(len(cities)))
        random.shuffle(initial_tour)
        initial_length = tour_length(initial_tour)

        sa = SimulatedAnnealing(
            objective_fn=tour_length,
            neighbor_fn=two_opt,
            initial_solution=initial_tour,
            initial_temp=5000.0,
            min_temp=1e-6,
            max_iterations=30_000,
            cooling_schedule=ExponentialCooling(alpha=0.9995),
        )
        best_tour, best_length, history = sa.optimize()

        improvement = (initial_length - best_length) / initial_length * 100

        return {
            "best_tour": best_tour,
            "best_length": best_length,
            "initial_length": initial_length,
            "improvement_pct": improvement,
            "iterations": len(history["cost"]),
        }

    # ------------------------------------------------------------------
    # NLP / sentiment
    # ------------------------------------------------------------------

    def solve_nlp_task(self, texts: List[str]) -> Dict[str, Any]:
        """Analyse the sentiment of a list of texts.

        Args:
            texts: Raw text strings to analyse.

        Returns:
            Dictionary with keys:

            - ``"results"`` — list of per-text sentiment dicts.
            - ``"summary"`` — counts of POSITIVE / NEGATIVE labels.
        """
        from python.nlp.sentiment_analysis import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        results = analyzer.batch_analyze(texts)

        summary: Dict[str, int] = {}
        for r in results:
            label = str(r["label"])
            summary[label] = summary.get(label, 0) + 1

        return {"results": results, "summary": summary}

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def auto_solve(self, problem_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch to the appropriate solver.

        Args:
            problem_type: One of ``"classification"``, ``"optimization"``,
                or ``"sentiment"``.
            **kwargs: Keyword arguments forwarded to the underlying solver.

        Returns:
            Solver-specific result dictionary.

        Raises:
            ValueError: If *problem_type* is not recognised.
        """
        dispatch = {
            "classification": self.solve_classification,
            "optimization": self.solve_optimization,
            "sentiment": self.solve_nlp_task,
        }
        if problem_type not in dispatch:
            raise ValueError(
                f"Unknown problem_type '{problem_type}'. "
                f"Choose from {list(dispatch.keys())}."
            )
        return dispatch[problem_type](**kwargs)
