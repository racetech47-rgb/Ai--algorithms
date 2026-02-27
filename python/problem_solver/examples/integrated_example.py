"""Integrated example demonstrating all ProblemSolver capabilities."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.problem_solver import ProblemSolver


# ── Sample data ──────────────────────────────────────────────────────────────

TRAIN_TEXTS = [
    "The movie was fantastic and I loved every minute",
    "Terrible film, total waste of time",
    "Absolutely brilliant performance by the cast",
    "Boring and predictable, fell asleep halfway through",
    "Outstanding special effects and gripping storyline",
    "Dull script, poor acting, not worth watching",
    "A masterpiece of modern cinema",
    "Disappointing sequel that ruins the original",
]
TRAIN_LABELS = [1, 0, 1, 0, 1, 0, 1, 0]

TEST_TEXTS = [
    "Wonderful experience, would watch again",
    "Dreadful movie, avoid at all costs",
]

SENTIMENT_TEXTS = [
    "I am so happy with this purchase!",
    "This is the worst product I have ever bought.",
    "It is okay, nothing extraordinary.",
]


def main() -> None:
    """Run all three solver types and print results."""
    solver = ProblemSolver()

    # ── Classification ───────────────────────────────────────────────────
    print("=" * 50)
    print("TEXT CLASSIFICATION")
    print("=" * 50)
    clf_result = solver.auto_solve(
        "classification",
        texts=TRAIN_TEXTS,
        labels=TRAIN_LABELS,
        test_texts=TEST_TEXTS,
    )
    print(f"Predictions : {clf_result['predictions']}")
    print("Train metrics:")
    for k, v in clf_result["metrics"].items():
        print(f"  {k:<12}: {v:.4f}")

    # ── Optimization ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TSP OPTIMIZATION (15 random cities)")
    print("=" * 50)
    opt_result = solver.auto_solve("optimization", num_cities=15)
    print(f"Initial tour length  : {opt_result['initial_length']:.2f}")
    print(f"Optimized tour length : {opt_result['best_length']:.2f}")
    print(f"Improvement          : {opt_result['improvement_pct']:.1f}%")
    print(f"Iterations           : {opt_result['iterations']}")
    # ── Sentiment ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS")
    print("=" * 50)
    nlp_result = solver.auto_solve("sentiment", texts=SENTIMENT_TEXTS)
    for text, res in zip(SENTIMENT_TEXTS, nlp_result["results"]):
        print(f"[{res['label']} {res['score']:.3f}]  {text}")
    print(f"Summary: {nlp_result['summary']}")


if __name__ == "__main__":
    main()
