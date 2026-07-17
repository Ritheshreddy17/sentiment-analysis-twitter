"""
preprocess.py
-------------
Cleans and normalizes raw tweet text so it can be fed into the model.

Steps:
1. Lowercase
2. Remove URLs, @mentions, retweet markers, and the '#' symbol (keep the word)
3. Tokenize
4. Remove stopwords and punctuation-only tokens
5. Lemmatize
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RETWEET_RE = re.compile(r"^rt\s+", flags=re.IGNORECASE)
_HASHTAG_SYMBOL_RE = re.compile(r"#")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def clean_tweet(text: str) -> str:
    """Lowercase + strip URLs/mentions/hashtag symbol/non-letters."""
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _RETWEET_RE.sub("", text)
    text = _HASHTAG_SYMBOL_RE.sub("", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    cleaned = clean_tweet(text)
    tokens = word_tokenize(cleaned)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and tok not in string.punctuation and len(tok) > 1
    ]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "RT @someone: I LOVE this new phone!!! Check it out http://example.com #amazing"
    print("Raw:      ", sample)
    print("Processed:", preprocess(sample))
