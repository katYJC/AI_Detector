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

def train_model():
    df = pd.read_csv("sample_data.csv")
    X_basic = np.array([extract_features(t) for t in df["text"]])
    y = df["label"].map({"Human": 0, "AI": 1})

    # 建立語言模型（用全部訓練文字）
    lm = SimpleLanguageModel(df["text"].tolist())

    # 加入 Perplexity / Burstiness
    extra_feats = []
    for t in df["text"]:
        extra_feats.append([
            lm.perplexity(t),
            burstiness(t)
        ])

    X = np.hstack([X_basic, np.array(extra_feats)])

    model = LogisticRegression()
    model.fit(X, y)

    return model, lm


import math
from collections import Counter
import re
import numpy as np

# ---------- Perplexity (Simple Unigram LM) ----------
class SimpleLanguageModel:
    def __init__(self, texts):
        tokens = []
        for t in texts:
            tokens.extend(self.tokenize(t))
        self.counts = Counter(tokens)
        self.total = sum(self.counts.values())
        self.vocab_size = len(self.counts)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def perplexity(self, text):
        tokens = self.tokenize(text)
        log_prob = 0.0
        for w in tokens:
            # Laplace smoothing
            prob = (self.counts.get(w, 0) + 1) / (self.total + self.vocab_size)
            log_prob += math.log(prob)
        return math.exp(-log_prob / max(len(tokens), 1))


# ---------- Burstiness ----------
def burstiness(text):
    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) <= 1:
        return 0.0
    return float(np.std(lengths))
