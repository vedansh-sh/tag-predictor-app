import os
import re
import warnings

import gradio as gr
import joblib
import nltk
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
nltk.download("stopwords", quiet=True)

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "models", "tag_predictor_model.pkl"))
tfidf = joblib.load(os.path.join(BASE, "models", "tfidf_vectorizer.pkl"))
mlb = joblib.load(os.path.join(BASE, "models", "multilabel_binarizer.pkl"))

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z0-9#+.\- ]", " ", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


def predict(title, body, top_k):
    if not title.strip() and not body.strip():
        return {}
    cleaned = clean_text(f"{title} {title} {body}")
    X = tfidf.transform([cleaned])
    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[::-1][: int(top_k)]
    return {mlb.classes_[i]: float(probs[i]) for i in top_idx}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Question title", placeholder="How to merge two dictionaries in Python?"),
        gr.Textbox(label="Question body", lines=8, placeholder="Paste the question body (HTML is fine)"),
        gr.Slider(1, 10, value=5, step=1, label="Number of tags to show"),
    ],
    outputs=gr.Label(label="Predicted tags (confidence)"),
    title="Stack Overflow Tag Predictor",
    description="Multi-label text classifier (TF-IDF + One-vs-Rest SGD) trained on 150k Stack Overflow questions across the top 500 tags.",
    examples=[[
        "How to merge two dictionaries in Python?",
        "<p>I have two dictionaries and want to combine them, with values from the second overwriting the first.</p>",
        5,
    ]],
)

if __name__ == "__main__":
    demo.launch()