# NLP

## Overview

The NLP module (Python only) provides three composable components:

| Component | Class | Purpose |
|-----------|-------|---------|
| Text Preprocessing | `TextPreprocessor` | Clean and normalize raw text |
| Sentiment Analysis | `SentimentAnalyzer` | Classify text as POSITIVE / NEGATIVE with a confidence score |
| Text Classification | `TextClassifier` | Multi-class document classification using TF-IDF + Logistic Regression |

All classes are in the `python.nlp` package and can be imported individually or together.

```python
from python.nlp import TextPreprocessor, SentimentAnalyzer, TextClassifier
```

---

## Text Preprocessing Pipeline

`TextPreprocessor` applies a deterministic, configurable pipeline to raw text before it is fed to a downstream model.

### Pipeline steps (in order)

1. Lower-case the input
2. Remove URLs (`http://…`, `https://…`, `www.…`)
3. Strip punctuation and digits
4. Tokenize with NLTK `word_tokenize`
5. Remove English stop-words *(optional)*
6. Lemmatize (WordNet) or stem (Porter) *(optional)*
7. Rejoin tokens into a single space-separated string

### Constructor

```python
from python.nlp.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    remove_stopwords=True,   # remove common English stop-words
    stemming=False,          # apply Porter stemming
    lemmatization=True,      # apply WordNet lemmatization (takes precedence over stemming)
)
```

> **Note:** NLTK resources (`punkt`, `stopwords`, `wordnet`) are downloaded automatically on first use if not already present.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `preprocess` | `(text: str) → str` | Process a single text; returns cleaned string |
| `tokenize` | `(text: str) → List[str]` | Process and return a list of tokens |
| `batch_preprocess` | `(texts: List[str]) → List[str]` | Process a list of texts |

### Example

```python
from python.nlp.preprocessing import TextPreprocessor

tp = TextPreprocessor()

print(tp.preprocess("The quick brown FOX jumped over the lazy dog!!"))
# quick brown fox jump lazy dog

print(tp.tokenize("Running quickly through the forest"))
# ['run', 'quickly', 'forest']

results = tp.batch_preprocess([
    "I love machine learning!",
    "Deep learning is amazing.",
])
# ['love machine learning', 'deep learning amazing']
```

---

## Sentiment Analysis

`SentimentAnalyzer` wraps the Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` model via the `transformers` pipeline. When `transformers` is not installed or the model cannot load, it falls back to a lightweight lexicon-based classifier.

### Constructor

```python
from python.nlp.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    model_name="distilbert-base-uncased-finetuned-sst-2-english"
)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `analyze` | `(text: str) → dict` | Returns `{"label": "POSITIVE"\|"NEGATIVE", "score": float}` |
| `batch_analyze` | `(texts: List[str]) → List[dict]` | Batch version of `analyze` |

### Single-text example

```python
from python.nlp.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("I absolutely love this product!")
# {"label": "POSITIVE", "score": 0.9998}
```

### Batch example

```python
texts = [
    "I absolutely love this product — it exceeded all my expectations!",
    "Terrible experience. The item broke after one day and support was unhelpful.",
    "Decent quality for the price. Nothing special but does the job.",
    "Best purchase I've made this year. Highly recommend to everyone!",
    "Awful. Would not buy again. Complete waste of money.",
]

results = analyzer.batch_analyze(texts)
for text, res in zip(texts, results):
    print(f"[{res['label']} {res['score']:.3f}]  {text[:65]}")
```

**Expected output**

```
[POSITIVE 1.000]  I absolutely love this product — it exceeded all my expe
[NEGATIVE 0.999]  Terrible experience. The item broke after one day and sup
[POSITIVE 0.512]  Decent quality for the price. Nothing special but does th
[POSITIVE 1.000]  Best purchase I've made this year. Highly recommend to ev
[NEGATIVE 0.999]  Awful. Would not buy again. Complete waste of money.
```

---

## Text Classification

`TextClassifier` implements a scikit-learn `Pipeline` combining:

- **TF-IDF vectorizer** — unigram + bigram features, max 50,000 vocabulary, sublinear TF scaling
- **Logistic Regression** — L-BFGS solver, up to 1000 iterations

### Constructor

```python
from python.nlp.text_classification import TextClassifier

clf = TextClassifier(num_classes=2)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `(texts: List[str], labels: List[int]) → TextClassifier` | Train on labelled data; returns self |
| `predict` | `(texts: List[str]) → List[int]` | Predict integer class labels |
| `predict_proba` | `(texts: List[str]) → np.ndarray` | Return class probability array (n_samples, n_classes) |
| `evaluate` | `(texts: List[str], labels: List[int]) → dict` | Returns `accuracy`, `precision`, `recall`, `f1` |

### Training and inference example

```python
from python.nlp.text_classification import TextClassifier

train_texts = [
    "The movie was fantastic and I loved every minute",
    "Terrible film, total waste of time",
    "Absolutely brilliant performance by the cast",
    "Boring and predictable, fell asleep halfway through",
    "Outstanding special effects and gripping storyline",
    "Dull script, poor acting, not worth watching",
]
train_labels = [1, 0, 1, 0, 1, 0]

clf = TextClassifier(num_classes=2)
clf.fit(train_texts, train_labels)

test_texts = [
    "Wonderful experience, would watch again",
    "Dreadful movie, avoid at all costs",
]
predictions = clf.predict(test_texts)
print(predictions)  # [1, 0]

proba = clf.predict_proba(test_texts)
print(proba)
# [[0.12, 0.88],
#  [0.93, 0.07]]
```

### Evaluation example

```python
metrics = clf.evaluate(train_texts, train_labels)
print(metrics)
# {
#   "accuracy" : 1.0,
#   "precision": 1.0,
#   "recall"   : 1.0,
#   "f1"       : 1.0,
# }
```

### Combining preprocessing with classification

```python
from python.nlp.preprocessing import TextPreprocessor
from python.nlp.text_classification import TextClassifier

tp  = TextPreprocessor()
clf = TextClassifier(num_classes=2)

processed_train = tp.batch_preprocess(train_texts)
clf.fit(processed_train, train_labels)

processed_test = tp.batch_preprocess(test_texts)
print(clf.predict(processed_test))  # [1, 0]
```
