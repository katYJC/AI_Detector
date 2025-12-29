import matplotlib
from matplotlib import font_manager, rcParams

# 指定中文字型路徑（相對路徑）
font_path = "fonts/NotoSansCJKtc-Regular.otf"
font_manager.fontManager.addfont(font_path)

font_prop = font_manager.FontProperties(fname=font_path)

rcParams["font.family"] = font_prop.get_name()
rcParams["axes.unicode_minus"] = False  # 解決負號顯示成方塊

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import extract_features, train_model

st.set_page_config(page_title="AI Detector", layout="centered")
st.title("🧠 AI vs Human 文章偵測器")

model = train_model()

# --- Optional: read training data for reference stats (if exists) ---
@st.cache_data
def load_train_stats():
    try:
        df = pd.read_csv("sample_data.csv")
        X = np.array([extract_features(t) for t in df["text"]])
        y = df["label"].map({"Human": 0, "AI": 1}).values
        feature_names = ["字數", "平均句長", "詞彙多樣性", "標點比例"]
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
    else:
        feature_names = ["字數", "平均句長", "詞彙多樣性", "標點比例"]
        features = np.array(extract_features(text)).reshape(1, -1)

        proba_ai = model.predict_proba(features)[0][1]
        pred = 1 if proba_ai >= 0.5 else 0

        # --- Result Header ---
        if pred == 1:
            st.error(f"🤖 判斷結果：AI 生成文章（AI 機率 {proba_ai:.2f}）")
        else:
            st.success(f"✍️ 判斷結果：人類撰寫文章（AI 機率 {proba_ai:.2f}）")

        # --- Visualization 1: Probability bar ---
        st.subheader("🎯 AI 機率（信心）")
        st.progress(float(proba_ai))
        st.caption("0 越像 Human，1 越像 AI（這是模型的機率輸出，不代表絕對正確）")

        # --- Feature table ---
        st.subheader("📊 特徵統計量")
        feat_dict = dict(zip(feature_names, features[0]))
        st.write({
            "字數": int(feat_dict["字數"]),
            "平均句長": round(float(feat_dict["平均句長"]), 2),
            "詞彙多樣性": round(float(feat_dict["詞彙多樣性"]), 2),
            "標點比例": round(float(feat_dict["標點比例"]), 3)
        })

        # --- Visualization 2: Feature bar chart ---
        st.subheader("📈 特徵長條圖")
        fig, ax = plt.subplots()
        ax.bar(feature_names, features[0])
        ax.set_ylabel("值")
        ax.set_title("輸入文字的特徵分佈")
        st.pyplot(fig)

        # --- Visualization 3 (Optional): Compare with training averages ---
        if train_stats is not None:
            st.subheader("🧭 與訓練資料平均值對照（Human vs AI）")

            # Build comparison dataframe
            compare_df = pd.DataFrame({
                "你的文字": features[0],
                "Human 平均": train_stats.loc[0].values,
                "AI 平均": train_stats.loc[1].values
            }, index=feature_names)

            st.dataframe(compare_df.style.format("{:.3f}"))

            # plot comparison (grouped bars)
            fig2, ax2 = plt.subplots()
            x = np.arange(len(feature_names))
            width = 0.25
            ax2.bar(x - width, compare_df["你的文字"].values, width, label="你的文字")
            ax2.bar(x, compare_df["Human 平均"].values, width, label="Human 平均")
            ax2.bar(x + width, compare_df["AI 平均"].values, width, label="AI 平均")
            ax2.set_xticks(x)
            ax2.set_xticklabels(feature_names)
            ax2.set_ylabel("值")
            ax2.set_title("特徵對照圖")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.info("找不到 sample_data.csv 或格式有誤：已略過訓練資料平均值對照圖。")

