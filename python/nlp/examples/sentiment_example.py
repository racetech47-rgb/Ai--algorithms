"""Sentiment analysis example using SentimentAnalyzer."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.nlp import SentimentAnalyzer


SAMPLE_TEXTS = [
    "I absolutely love this product — it exceeded all my expectations!",
    "Terrible experience. The item broke after one day and support was unhelpful.",
    "Decent quality for the price. Nothing special but does the job.",
    "Best purchase I've made this year. Highly recommend to everyone!",
    "Awful. Would not buy again. Complete waste of money.",
]


def main() -> None:
    """Run batch sentiment analysis and print labelled results."""
    analyzer = SentimentAnalyzer()
    print("Sentiment Analysis Results\n" + "=" * 40)
    results = analyzer.batch_analyze(SAMPLE_TEXTS)
    for text, result in zip(SAMPLE_TEXTS, results):
        label = result["label"]
        score = result["score"]
        print(f"[{label} {score:.3f}]  {text[:70]}")


if __name__ == "__main__":
    main()
