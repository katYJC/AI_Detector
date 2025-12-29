import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -------- Feature Engineering --------
def extract_features(text):
    words = text.split()
    sentences = re.split(r'[.!?]', text)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sentence_length = word_count / sentence_count
    unique_word_ratio = len(set(words)) / max(word_count, 1)
    punctuation_ratio = sum(1 for c in text if c in ".,!?") / max(len(text), 1)

    return [
        word_count,
        avg_sentence_length,
        unique_word_ratio,
        punctuation_ratio
    ]

# -------- Train Model --------
def train_model():
    df = pd.read_csv("sample_data.csv")
    X = np.array([extract_features(t) for t in df["text"]])
    y = df["label"].map({"Human": 0, "AI": 1})

    model = LogisticRegression()
    model.fit(X, y)
    return model

