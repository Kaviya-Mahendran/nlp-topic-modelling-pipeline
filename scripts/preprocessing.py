import re
import pandas as pd
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (safe to call multiple times)
nltk.download("stopwords")
nltk.download("wordnet")


class TextPreprocessor:
    """
    Cleans and normalises raw text for NLP pipelines.
    Designed for real-world CRM, feedback, or survey text.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _remove_pii(self, text: str) -> str:
        """
        Removes obvious personally identifiable patterns.
        """
        text = re.sub(r"\S+@\S+", "", text)  # emails
        text = re.sub(r"\b\d+\b", "", text)  # standalone numbers
        return text

    def clean_text(self, text: Optional[str]) -> str:
        """
        Main text cleaning function.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self._remove_pii(text)
        text = re.sub(r"[^a-z\s]", "", text)

        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]

        return " ".join(tokens)

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        """
        Applies preprocessing to a pandas Series.
        """
        return series.apply(self.clean_text)
