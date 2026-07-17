"""
predict.py
----------
Loads the trained model and classifies new/unseen tweets as
Positive, Negative, or Neutral.

Note on the Neutral class:
The NLTK twitter_samples corpus only contains Positive and Negative
examples, so the model itself is trained as a binary classifier.
To produce a 3-way Positive / Neutral / Negative output (as required
by the project brief), we add a confidence threshold: if the model
isn't at least NEUTRAL_THRESHOLD confident in either class, the tweet
is labeled Neutral. This is a common, practical technique used when a
true 3-class labeled dataset isn't available.

Usage:
    python predict.py "I absolutely loved this new update!"
    python predict.py            # runs on a few built-in example tweets
"""

import sys
import joblib
from preprocess import preprocess

NEUTRAL_THRESHOLD = 0.60  # below this confidence on both classes -> Neutral

MODEL_PATH = "model/sentiment_model.joblib"
VECTORIZER_PATH = "model/vectorizer.joblib"

_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def classify(text: str):
    model, vectorizer = _load()

    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])

    proba = model.predict_proba(vec)[0]
    classes = model.classes_  # ['negative', 'positive']
    class_proba = dict(zip(classes, proba))

    top_label = max(class_proba, key=class_proba.get)
    top_conf = class_proba[top_label]

    if top_conf < NEUTRAL_THRESHOLD:
        label = "neutral"
    else:
        label = top_label

    return {
        "text": text,
        "cleaned_text": cleaned,
        "label": label,
        "confidence": round(top_conf, 3),
        "raw_probabilities": {k: round(v, 3) for k, v in class_proba.items()},
    }


SAMPLE_TWEETS = [
    "I absolutely love the new update, it's fantastic!",
    "This is the worst service I have ever experienced.",
    "The event starts at 6pm tomorrow at the main hall.",
    "Not sure how I feel about this, could go either way.",
    "Best purchase I've made all year, highly recommend!",
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tweet = " ".join(sys.argv[1:])
        result = classify(tweet)
        print(f"\nTweet:      {result['text']}")
        print(f"Sentiment:  {result['label'].upper()}  (confidence: {result['confidence']})")
    else:
        print("No tweet given as an argument, running on sample tweets:\n")
        for t in SAMPLE_TWEETS:
            result = classify(t)
            print(f"- \"{t}\"\n  -> {result['label'].upper()} (confidence: {result['confidence']})\n")
