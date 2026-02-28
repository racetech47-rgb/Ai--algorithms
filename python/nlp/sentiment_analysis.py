"""Sentiment analysis using a Hugging Face transformer pipeline with
a simple lexicon-based fallback when transformers is unavailable."""

from __future__ import annotations

from typing import Dict, List


# ── Lexicon-based fallback ───────────────────────────────────────────────────

_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "best", "happy", "awesome", "superb", "brilliant", "enjoy",
    "perfect", "outstanding", "positive", "beautiful", "nice", "like",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "worst", "hate", "poor",
    "disappointing", "disgusting", "negative", "ugly", "boring", "sad",
    "dislike", "dreadful", "mediocre", "failure", "wrong", "weak",
}


def _lexicon_sentiment(text: str) -> Dict[str, object]:
    words = set(text.lower().split())
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    if pos_count > neg_count:
        label, score = "POSITIVE", min(0.5 + 0.1 * pos_count, 0.99)
    elif neg_count > pos_count:
        label, score = "NEGATIVE", min(0.5 + 0.1 * neg_count, 0.99)
    else:
        label, score = "POSITIVE", 0.5
    return {"label": label, "score": float(score)}


# ── Main class ───────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """Analyse the sentiment of text using a pre-trained transformer model.

    Falls back to a lightweight lexicon-based classifier when the
    ``transformers`` library is not installed or the model cannot be loaded.

    Args:
        model_name: Hugging Face model identifier.

    Example:
        >>> sa = SentimentAnalyzer()
        >>> sa.analyze("I love this product!")
        {'label': 'POSITIVE', 'score': 0.9998...}
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ) -> None:
        self._model_name = model_name
        self._pipeline = None
        self._use_fallback = False
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                truncation=True,
                max_length=512,
            )
        except Exception:
            self._use_fallback = True

    def analyze(self, text: str) -> Dict[str, object]:
        """Analyse the sentiment of a single text.

        Args:
            text: Input text string.

        Returns:
            Dictionary with keys ``'label'`` (``'POSITIVE'`` or
            ``'NEGATIVE'``) and ``'score'`` (confidence in [0, 1]).
        """
        if self._use_fallback or self._pipeline is None:
            return _lexicon_sentiment(text)
        result = self._pipeline(text)[0]
        return {"label": result["label"], "score": float(result["score"])}

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, object]]:
        """Analyse the sentiment of multiple texts.

        Args:
            texts: List of input text strings.

        Returns:
            List of sentiment dictionaries (see :meth:`analyze`).
        """
        if self._use_fallback or self._pipeline is None:
            return [_lexicon_sentiment(t) for t in texts]
        results = self._pipeline(texts)
        return [{"label": r["label"], "score": float(r["score"])} for r in results]
