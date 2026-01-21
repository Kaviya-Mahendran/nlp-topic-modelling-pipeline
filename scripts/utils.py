from textblob import TextBlob


def get_sentiment(text: str) -> float:
    """
    Returns sentiment polarity score between -1 and 1.
    Negative values indicate negative sentiment,
    positive values indicate positive sentiment.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    return TextBlob(text).sentiment.polarity
