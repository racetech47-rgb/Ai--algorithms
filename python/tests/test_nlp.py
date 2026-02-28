"""Unit tests for the nlp module."""

from __future__ import annotations

import unittest

from python.nlp.preprocessing import TextPreprocessor
from python.nlp.text_classification import TextClassifier


class TestTextPreprocessor(unittest.TestCase):
    """Tests for TextPreprocessor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.preprocessor = TextPreprocessor(
            remove_stopwords=True,
            stemming=False,
            lemmatization=True,
        )

    def test_preprocess_returns_string(self) -> None:
        result = self.preprocessor.preprocess("Hello World!")
        self.assertIsInstance(result, str)

    def test_preprocess_lowercases(self) -> None:
        result = self.preprocessor.preprocess("HELLO WORLD")
        self.assertEqual(result, result.lower())

    def test_preprocess_removes_punctuation(self) -> None:
        result = self.preprocessor.preprocess("Hello, world!!!")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)

    def test_preprocess_removes_urls(self) -> None:
        result = self.preprocessor.preprocess("Visit https://example.com today")
        self.assertNotIn("https", result)
        self.assertNotIn("example", result)

    def test_preprocess_empty_string(self) -> None:
        result = self.preprocessor.preprocess("")
        self.assertEqual(result, "")

    def test_tokenize_returns_list(self) -> None:
        tokens = self.preprocessor.tokenize("The quick brown fox")
        self.assertIsInstance(tokens, list)

    def test_tokenize_non_empty(self) -> None:
        tokens = self.preprocessor.tokenize("Running quickly through the forest")
        self.assertGreater(len(tokens), 0)

    def test_tokenize_elements_are_strings(self) -> None:
        tokens = self.preprocessor.tokenize("Testing one two three")
        for token in tokens:
            self.assertIsInstance(token, str)

    def test_batch_preprocess_returns_list(self) -> None:
        texts = ["Hello world", "Foo bar baz"]
        result = self.preprocessor.batch_preprocess(texts)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_batch_preprocess_all_strings(self) -> None:
        texts = ["Hello world", "Foo bar baz", ""]
        result = self.preprocessor.batch_preprocess(texts)
        for item in result:
            self.assertIsInstance(item, str)

    def test_no_stopwords_option(self) -> None:
        tp = TextPreprocessor(remove_stopwords=False, lemmatization=False)
        result = tp.preprocess("the cat sat on the mat")
        self.assertIn("the", result)

    def test_stemming_option(self) -> None:
        tp = TextPreprocessor(remove_stopwords=False, stemming=True, lemmatization=False)
        result = tp.preprocess("running quickly")
        self.assertIsInstance(result, str)


class TestTextClassifier(unittest.TestCase):
    """Tests for TextClassifier."""

    _TEXTS = [
        "great product love it",
        "terrible waste money",
        "excellent quality highly recommend",
        "awful experience never again",
        "best purchase happy",
        "horrible product broken",
        "amazing value outstanding",
        "disappointing poor quality",
    ]
    _LABELS = [1, 0, 1, 0, 1, 0, 1, 0]

    @classmethod
    def setUpClass(cls) -> None:
        cls.clf = TextClassifier(num_classes=2)
        cls.clf.fit(cls._TEXTS, cls._LABELS)

    def test_predict_returns_list(self) -> None:
        result = self.clf.predict(["great product"])
        self.assertIsInstance(result, list)

    def test_predict_correct_length(self) -> None:
        texts = ["great product", "terrible item", "amazing deal"]
        result = self.clf.predict(texts)
        self.assertEqual(len(result), 3)

    def test_predict_valid_labels(self) -> None:
        result = self.clf.predict(self._TEXTS)
        for label in result:
            self.assertIn(label, [0, 1])

    def test_predict_proba_shape(self) -> None:
        import numpy as np

        proba = self.clf.predict_proba(self._TEXTS)
        self.assertEqual(proba.shape, (len(self._TEXTS), 2))

    def test_predict_proba_sums_to_one(self) -> None:
        import numpy as np

        proba = self.clf.predict_proba(self._TEXTS)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_evaluate_returns_expected_keys(self) -> None:
        metrics = self.clf.evaluate(self._TEXTS, self._LABELS)
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(key, metrics)

    def test_evaluate_accuracy_in_range(self) -> None:
        metrics = self.clf.evaluate(self._TEXTS, self._LABELS)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_predict_before_fit_raises(self) -> None:
        clf = TextClassifier(num_classes=2)
        with self.assertRaises(RuntimeError):
            clf.predict(["test"])

    def test_fit_returns_self(self) -> None:
        clf = TextClassifier(num_classes=2)
        result = clf.fit(self._TEXTS, self._LABELS)
        self.assertIs(result, clf)


if __name__ == "__main__":
    unittest.main()
