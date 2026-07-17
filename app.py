"""
app.py
------
A simple web UI for the sentiment analysis model, built with Streamlit.

Run with:
    streamlit run app.py

Needs model/sentiment_model.joblib and model/vectorizer.joblib to already
exist - run prepare_data.py and train_model.py first if you haven't.
"""

import streamlit as st
from predict import classify

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦", layout="centered")

st.title("🐦 Twitter Sentiment Analyzer")
st.caption("AI Internship Project — Codec Technologies | Yerraguntla Rithesh Reddy")

st.write(
    "Type any sentence below (as if it were a tweet) and the model will "
    "classify it as **Positive**, **Negative**, or **Neutral**."
)

text = st.text_area("Enter a tweet / sentence:", height=100, placeholder="e.g. I absolutely love this new update!")

col1, col2 = st.columns([1, 3])
with col1:
    analyze = st.button("Analyze Sentiment", type="primary")

if analyze:
    if not text.strip():
        st.warning("Please type something first.")
    else:
        result = classify(text)
        label = result["label"]
        conf = result["confidence"]

        st.markdown("---")
        if label == "positive":
            st.success(f"### 😊 POSITIVE")
        elif label == "negative":
            st.error(f"### 😠 NEGATIVE")
        else:
            st.info(f"### 😐 NEUTRAL")

        st.write(f"**Confidence:** {conf * 100:.1f}%")
        st.progress(conf)

        with st.expander("See raw model output"):
            st.json(result)

st.markdown("---")
st.caption("Try these examples:")
examples = [
    "I absolutely love this new update!",
    "This is the worst experience ever.",
    "The meeting has been moved to 3pm",
]
ex_cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if ex_cols[i].button(ex, key=f"ex_{i}"):
        result = classify(ex)
        st.write(f"**\"{ex}\"** → **{result['label'].upper()}** (confidence: {result['confidence']*100:.1f}%)")

with st.sidebar:
    st.header("Model Info")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** TF-IDF (unigrams + bigrams)")
    st.write("**Training data:** 10,000 real tweets (NLTK twitter_samples)")
    st.write("**Test accuracy:** ~74.8%")
    st.write("**Neutral rule:** confidence below 60% on both classes")