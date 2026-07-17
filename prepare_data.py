"""
prepare_data.py
----------------
Loads the built-in NLTK 'twitter_samples' corpus (5,000 positive and
5,000 negative real tweets) and saves it as a single CSV file that the
rest of the project works from.

Run this once before train_model.py.
"""

import pandas as pd
import nltk
from nltk.corpus import twitter_samples

REQUIRED = ["twitter_samples", "stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]


def ensure_nltk_data():
    for pkg in REQUIRED:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg)


def build_dataset(out_path="data/tweets.csv"):
    ensure_nltk_data()

    pos = twitter_samples.strings("positive_tweets.json")
    neg = twitter_samples.strings("negative_tweets.json")

    rows = [{"text": t, "label": "positive"} for t in pos]
    rows += [{"text": t, "label": "negative"} for t in neg]

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} labeled tweets to {out_path}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    build_dataset()
