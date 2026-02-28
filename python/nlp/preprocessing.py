"""Text preprocessing utilities using NLTK."""

from __future__ import annotations

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    for name, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


class TextPreprocessor:
    """Preprocess raw text for NLP tasks.

    Args:
        remove_stopwords: Whether to remove common English stop-words.
        stemming: Whether to apply Porter stemming.
        lemmatization: Whether to apply WordNet lemmatization (takes
            precedence over stemming when both are True).

    Example:
        >>> tp = TextPreprocessor()
        >>> tp.preprocess("The quick brown FOX jumped!!")
        'quick brown fox jump'
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        stemming: bool = False,
        lemmatization: bool = True,
    ) -> None:
        _ensure_nltk_resources()
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self._stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if lemmatization else None
        self._stemmer = PorterStemmer() if stemming else None

    def preprocess(self, text: str) -> str:
        """Clean and normalise a single text string.

        Steps applied in order:
        1. Lower-case
        2. Remove URLs
        3. Remove punctuation and digits
        4. Tokenize
        5. Remove stop-words (optional)
        6. Lemmatize or stem (optional)
        7. Rejoin tokens

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text as a single space-separated string.
        """
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
        tokens = word_tokenize(text)

        processed: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if self.remove_stopwords and token in self._stop_words:
                continue
            if self.lemmatization and self._lemmatizer:
                token = self._lemmatizer.lemmatize(token)
            elif self.stemming and self._stemmer:
                token = self._stemmer.stem(token)
            processed.append(token)

        return " ".join(processed)

    def tokenize(self, text: str) -> List[str]:
        """Preprocess and tokenize text into a list of tokens.

        Args:
            text: Raw input text.

        Returns:
            List of preprocessed tokens.
        """
        return self.preprocess(text).split()

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of preprocessed text strings.
        """
        return [self.preprocess(t) for t in texts]
