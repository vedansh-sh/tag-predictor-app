import os
import re
import warnings

import joblib
import nltk
import streamlit as st
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
nltk.download("stopwords", quiet=True)

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, "models", "tag_predictor_model.pkl"))
    tfidf = joblib.load(os.path.join(BASE, "models", "tfidf_vectorizer.pkl"))
    mlb = joblib.load(os.path.join(BASE, "models", "multilabel_binarizer.pkl"))
    return model, tfidf, mlb


model, tfidf, mlb = load_artifacts()
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z0-9#+.\- ]", " ", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


st.set_page_config(page_title="Stack Overflow Tag Predictor", layout="centered")
st.title("Stack Overflow Tag Predictor")
st.caption("Multi-label classifier (TF-IDF + One-vs-Rest SGD) trained on 150k Stack Overflow questions across the top 500 tags.")

title = st.text_input("Question title", placeholder="How to merge two dictionaries in Python?")
body = st.text_area("Question body", height=200, placeholder="Paste the question body (HTML is fine)")
top_k = st.slider("Number of tags to show", 1, 10, 5)

if st.button("Predict tags"):
    if not title.strip() and not body.strip():
        st.warning("Enter a title or body first.")
    else:
        cleaned = clean_text(f"{title} {title} {body}")
        X = tfidf.transform([cleaned])
        probs = model.predict_proba(X)[0]
        top_idx = probs.argsort()[::-1][:top_k]
        st.subheader("Predicted tags")
        for i in top_idx:
            st.write(f"{mlb.classes_[i]}  -  {probs[i]:.2f}")
            st.progress(min(float(probs[i]), 1.0))
