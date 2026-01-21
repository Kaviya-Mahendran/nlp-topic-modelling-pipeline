import pandas as pd
from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib


class TopicModel:
    """
    Trains and applies an LDA topic model on cleaned text.
    Designed for interpretability over complexity.
    """

    def __init__(
        self,
        n_topics: int = 5,
        max_features: int = 1000,
        random_state: int = 42
    ):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english"
        )
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state
        )

    def fit(self, texts: List[str]) -> None:
        """
        Fits the vectorizer and LDA model.
        """
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(self.doc_term_matrix)

    def transform(self, texts: List[str]) -> List[int]:
        """
        Assigns dominant topic to each document.
        """
        dtm = self.vectorizer.transform(texts)
        topic_distributions = self.model.transform(dtm)
        return topic_distributions.argmax(axis=1)

    def get_topics(self, n_words: int = 8) -> List[Tuple[int, List[str]]]:
        """
        Returns top keywords for each topic.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.model.components_):
            top_words = [
                feature_names[i]
                for i in topic.argsort()[:-n_words - 1:-1]
            ]
            topics.append((topic_idx, top_words))

        return topics

    def save(self, path: str) -> None:
        """
        Saves model and vectorizer.
        """
        joblib.dump(
            {
                "model": self.model,
                "vectorizer": self.vectorizer
            },
            path
        )
