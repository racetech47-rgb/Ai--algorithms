# NLP

## Overview

The NLP module provides three composable components:

| Component | Class | Purpose |
|-----------|-------|---------|
| Text Preprocessing | `TextPreprocessor` | Clean and normalize raw text |
| Sentiment Analysis | `SentimentAnalyzer` | Classify text as POSITIVE / NEGATIVE |
| Text Classification | `TextClassifier` | Multi-class document classification |

```python
from python.nlp import TextPreprocessor, SentimentAnalyzer, TextClassifier
```

---

## Text Preprocessing

```python
from python.nlp.preprocessing import TextPreprocessor

tp = TextPreprocessor(
    remove_stopwords=True,
    stemming=False,
    lemmatization=True,
)

print(tp.preprocess("The quick brown FOX jumped!!"))
# quick brown fox jump
```

**Methods:** `preprocess(text)`, `tokenize(text)`, `batch_preprocess(texts)`

---

## Sentiment Analysis

```python
from python.nlp.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("I love this product!")
# {"label": "POSITIVE", "score": 0.9998}
```

Falls back to a lexicon-based classifier when `transformers` is unavailable.

**Methods:** `analyze(text)`, `batch_analyze(texts)`

---

## Text Classification

```python
from python.nlp.text_classification import TextClassifier

clf = TextClassifier(num_classes=2)
clf.fit(train_texts, train_labels)
predictions = clf.predict(test_texts)
metrics = clf.evaluate(test_texts, test_labels)
```

**Methods:** `fit(texts, labels)`, `predict(texts)`, `predict_proba(texts)`, `evaluate(texts, labels)`
