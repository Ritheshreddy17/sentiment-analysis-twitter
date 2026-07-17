"""
train_model.py
---------------
Trains a sentiment classifier on the prepared tweet dataset.

Pipeline:
    raw tweet -> preprocess.py -> TF-IDF features -> Logistic Regression

Outputs (saved into model/):
    sentiment_model.joblib   - trained classifier
    vectorizer.joblib        - fitted TF-IDF vectorizer
    confusion_matrix.png     - evaluation plot
    sentiment_distribution.png - class balance plot

Run: python train_model.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

from preprocess import preprocess

DATA_PATH = "data/tweets.csv"
MODEL_DIR = "model"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning & preprocessing text (this takes a moment)...")
    df["clean_text"] = df["text"].astype(str).apply(preprocess)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    # --- Sentiment distribution plot (raw dataset) ---
    plt.figure(figsize=(5, 4))
    df["label"].value_counts().plot(kind="bar", color=["#1F3864", "#8496B0"])
    plt.title("Tweet Sentiment Distribution (Training Data)")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "sentiment_distribution.png"), dpi=150)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on held-out test set: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(classification_report(y_test, y_pred))

    print(f"\nSaved model, vectorizer, and evaluation plots to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
