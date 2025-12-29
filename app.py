import streamlit as st
import numpy as np
from model import extract_features, train_model

st.set_page_config(page_title="AI Detector", layout="centered")

st.title("🧠 AI vs Human 文章偵測器")

model = train_model()

text = st.text_area("請輸入一段文字：", height=200)

if st.button("開始判斷"):
    if text.strip() == "":
        st.warning("請先輸入文字")
    else:
        features = np.array(extract_features(text)).reshape(1, -1)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][prediction]

        if prediction == 1:
            st.error(f"🤖 判斷結果：AI 生成文章（信心 {prob:.2f}）")
        else:
            st.success(f"✍️ 判斷結果：人類撰寫文章（信心 {prob:.2f}）")

        st.subheader("📊 特徵統計")
        st.write({
            "字數": features[0][0],
            "平均句長": round(features[0][1], 2),
            "詞彙多樣性": round(features[0][2], 2),
            "標點比例": round(features[0][3], 3)
        })

