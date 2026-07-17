# Sentiment Analysis on Twitter Data

A simple, real, end-to-end NLP pipeline that classifies tweets as
**Positive**, **Negative**, or **Neutral**, built for the internship
project at Codec Technologies.

## How it works

1. **Data** — `prepare_data.py` pulls 10,000 real, labeled tweets
   (5,000 positive + 5,000 negative) from NLTK's built-in
   `twitter_samples` corpus and saves them to `data/tweets.csv`.
2. **Preprocessing** — `preprocess.py` cleans each tweet: lowercases
   it, strips URLs/@mentions/hashtag symbols, tokenizes it, removes
   stopwords, and lemmatizes the remaining words.
3. **Features** — `train_model.py` converts the cleaned text into
   TF-IDF vectors (unigrams + bigrams, top 5,000 terms).
4. **Model** — a Logistic Regression classifier is trained on an
   80/20 train/test split.
5. **Neutral class** — the raw dataset only has Positive/Negative
   examples, so `predict.py` adds a **confidence threshold**: if the
   model isn't at least 60% confident in either class, the tweet is
   labeled Neutral. This is a standard technique used in practice
   when a true 3-class labeled dataset isn't available — and it's
   worth mentioning this design choice if you're asked about it.

## Project structure

```
twitter_sentiment_project/
├── requirements.txt
├── prepare_data.py       # builds data/tweets.csv from NLTK twitter_samples
├── preprocess.py         # text cleaning + tokenizing + lemmatizing
├── train_model.py        # trains + evaluates + saves the model
├── predict.py            # classifies new tweets (Positive/Negative/Neutral)
├── data/
│   └── tweets.csv        # generated dataset (10,000 labeled tweets)
└── model/
    ├── sentiment_model.joblib
    ├── vectorizer.joblib
    ├── metrics.txt
    ├── confusion_matrix.png
    └── sentiment_distribution.png
```

## Setup

```bash
pip install -r requirements.txt
```

The first run will also download a few small NLTK data packages
(`twitter_samples`, `stopwords`, `punkt`, `wordnet`) automatically.

## Running it

```bash
# 1. Build the dataset (only needs to be run once)
python prepare_data.py

# 2. Train the model
python train_model.py

# 3. Try it on your own text
python predict.py "I absolutely loved this new update!"

# or run it on a few built-in example tweets
python predict.py
```

## Actual results from this run

- **Test accuracy: 74.84%** on a held-out 20% split of the 10,000-tweet dataset
- Precision/recall are balanced across both classes (~0.73–0.77)
- Full classification report is saved in `model/metrics.txt`
- `model/confusion_matrix.png` and `model/sentiment_distribution.png`
  are ready to drop straight into your presentation

## Honest limitations (good talking points for Q&A)

- The base dataset has no true Neutral-labeled tweets, so Neutral is
  inferred from low model confidence rather than learned directly —
  a real limitation worth being upfront about.
- Sarcasm and mixed sentiment within a single tweet are hard for a
  TF-IDF + Logistic Regression model to catch, since it has no sense
  of context beyond word/phrase frequency.
- Accuracy could likely be improved with a larger, hand-labeled
  3-class dataset, or a more advanced model (e.g. an LSTM or a
  transformer-based model like BERT), but that was outside the scope
  of a 2-month internship project.
