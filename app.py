import os
import urllib.request
from matplotlib import font_manager, rcParams

FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = "NotoSansCJKtc-Regular.otf"

if not os.path.exists(FONT_PATH):
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)

font_manager.fontManager.addfont(FONT_PATH)
rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
rcParams["axes.unicode_minus"] = False

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import (
    extract_features,
    burstiness,
    train_model
)

st.set_page_config(page_title="AI Detector", layout="centered")
st.title("🧠 AI vs Human 文章偵測器")

# ✅ 注意：train_model() 會回傳 model, lm
model, lm = train_model()

# --- Optional: read training data for reference stats (if exists) ---
@st.cache_data
def load_train_stats():
    try:
        df = pd.read_csv("sample_data.csv")

        feature_names = ["字數", "平均句長", "詞彙多樣性", "標點比例", "Perplexity", "Burstiness"]

        # 用同一套方法算訓練資料的 6 特徵
        X = []
        for t in df["text"]:
            basic = extract_features(t)
            ppl = lm.perplexity(t)
            brs = burstiness(t)
            X.append(basic + [ppl, brs])

        y = df["label"].map({"Human": 0, "AI": 1}).values

        Xdf = pd.DataFrame(X, columns=feature_names)
        Xdf["label"] = y
        stats = Xdf.groupby("label")[feature_names].mean()
        return stats  # index: 0=Human, 1=AI
    except Exception:
        return None

train_stats = load_train_stats()

text = st.text_area("請輸入一段文字：", height=200)

if st.button("開始判斷"):
    if text.strip() == "":
        st.warning("請先輸入文字")
        st.stop()

    # ✅ 算 6 個特徵（和模型訓練一致）
    basic = extract_features(text)
    ppl = lm.perplexity(text)
    burst = burstiness(text)

    feature_names = ["字數", "平均句長", "詞彙多樣性", "標點比例", "Perplexity", "Burstiness"]
    X = np.array(basic + [ppl, burst]).reshape(1, -1)

    proba_ai = float(model.predict_proba(X)[0][1])
    pred = 1 if proba_ai >= 0.5 else 0

    # --- Result Header ---
    if pred == 1:
        st.error(f"🤖 判斷結果：AI 生成文章（AI 機率 {proba_ai:.2f}）")
    else:
        st.success(f"✍️ 判斷結果：人類撰寫文章（AI 機率 {proba_ai:.2f}）")

    # --- Visualization 1: Probability bar ---
    st.subheader("🎯 AI 機率（信心）")
    st.progress(proba_ai)
    st.caption("0 越像 Human，1 越像 AI（這是模型的機率輸出，不代表絕對正確）")

    # --- Metrics: Perplexity & Burstiness ---
    st.subheader("🧠 語言風格指標")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Perplexity（困惑度）", f"{ppl:.2f}", help="越高代表越難預測，較像人類")
    with c2:
        st.metric("Burstiness（句長變化）", f"{burst:.2f}", help="句長變化越大，越偏人類")

    # --- Feature table ---
    st.subheader("📊 特徵統計量")
    feat_values = (basic + [ppl, burst])
    feat_dict = dict(zip(feature_names, feat_values))
    st.write({
        "字數": int(feat_dict["字數"]),
        "平均句長": round(float(feat_dict["平均句長"]), 2),
        "詞彙多樣性": round(float(feat_dict["詞彙多樣性"]), 2),
        "標點比例": round(float(feat_dict["標點比例"]), 3),
        "Perplexity": round(float(feat_dict["Perplexity"]), 2),
        "Burstiness": round(float(feat_dict["Burstiness"]), 2),
    })

    # --- Visualization 2: Feature bar chart (6 features) ---
    st.subheader("📈 特徵長條圖")
    fig, ax = plt.subplots()
    ax.bar(feature_names, feat_values)
    ax.set_ylabel("值")
    ax.set_title("輸入文字的特徵分佈")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

    # --- Visualization 3: Compare with training averages ---
    if train_stats is not None:
        st.subheader("🧭 與訓練資料平均值對照（Human vs AI）")

        compare_df = pd.DataFrame({
            "你的文字": feat_values,
            "Human 平均": train_stats.loc[0].values,
            "AI 平均": train_stats.loc[1].values
        }, index=feature_names)

        st.dataframe(compare_df.style.format("{:.3f}"))

        fig2, ax2 = plt.subplots()
        x = np.arange(len(feature_names))
        width = 0.25
        ax2.bar(x - width, compare_df["你的文字"].values, width, label="你的文字")
        ax2.bar(x, compare_df["Human 平均"].values, width, label="Human 平均")
        ax2.bar(x + width, compare_df["AI 平均"].values, width, label="AI 平均")
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names, rotation=20)
        ax2.set_ylabel("值")
        ax2.set_title("特徵對照圖")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("找不到 sample_data.csv 或格式有誤：已略過訓練資料平均值對照圖。")
