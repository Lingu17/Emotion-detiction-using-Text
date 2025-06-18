import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# 🚨 MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Emotion Detector", layout="centered")

# --- Custom CSS for better look ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0e7ff 0%, #f0fff4 100%);
    }
    .main {
        background-color: #ffffffcc;
        border-radius: 18px;
        padding: 2rem 2rem 1rem 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.12);
    }
    .stTextArea textarea {
        background: #f0f4f8;
        border-radius: 10px;
        font-size: 1.1rem;
        min-height: 120px;
        color: #000;
    }
    .stButton button {
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5em 2em;
        margin-top: 0.5em;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #06b6d4 0%, #6366f1 100%);
        color: #fff;
    }
    h1.emotion-title {
        font-family: "Brush Script MT", "Brush Script Std", cursive;
        font-size: 3.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown(
    "<h1 class='emotion-title' style='text-align:center; color:#6366f1; margin-bottom:0;'>Emotion Detector</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center; color:#64748b; font-size:1.1rem;'>Analyze the emotions in your text instantly!</p>", unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_classifier = load_emotion_model()

# --- UI ---
user_text = st.text_area("Type a sentence to analyze emotions:", height=120)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            predictions = emotion_classifier(user_text)[0]
            df = pd.DataFrame(predictions)
            df["score"] = df["score"].round(4)
            df = df.sort_values(by="score", ascending=False)

            # Add emoji mapping
            emoji_map = {
                "anger": "😠",
                "disgust": "🤢",
                "fear": "😨",
                "joy": "😄",
                "neutral": "😐",
                "sadness": "😢",
                "surprise": "😲",
                # Add more if your model supports more emotions
            }
            df["Emoji"] = df["label"].map(emoji_map).fillna("❓")

            st.subheader("Predicted Emotions")
            st.dataframe(
                df.rename(columns={"label": "Emotion", "score": "Confidence"})[["Emoji", "Emotion", "Confidence"]],
                use_container_width=True
            )

            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("score:Q", title="Confidence"),
                y=alt.Y("label:N", title="Emotion", sort='-x'),
                color=alt.Color("label:N", legend=None)
            ).properties(width=600, height=300)

            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- About Me & App Info ---
st.markdown(
    """
    <div style='
        max-width: 600px;
        margin: 40px auto 24px auto;
        background: #f8fafc;
        border-radius: 18px;
        border: 2px solid #6366f1;
        box-shadow: 0 4px 24px rgba(99,102,241,0.08);
        padding: 28px 32px 18px 32px;
        text-align: center;
    '>
        <h3 style='color:#6366f1; margin-bottom:0.5em;'>About This App</h3>
        <p style='color:#374151; font-size:1.08rem; margin-bottom:1.2em;'>
            <strong>Emotion Detector</strong> is an AI-powered web app that analyzes the emotions in your text using a state-of-the-art transformer model.<br>
            Enter any sentence and instantly see the predicted emotions and their confidence scores.
        </p>
        <h4 style='color:#6366f1; margin-bottom:0.2em; margin-top:1.5em;'>About Me</h4>
        <p style='color:#374151; font-size:1.05rem;'>
            Hi, I'm Lingraj! I love building AI-powered apps that help people understand language and emotions.<br>
            Connect with me on <a href="mailto:lingrajmalipatil17@email.com" style="color:#06b6d4;">email</a>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Footer ---
st.markdown(
    "<hr><div style='text-align:center; color:#a0aec0; font-size:0.95rem;'>"
    "&copy; 2025 Emotion Detector App"
    "</div>",
    unsafe_allow_html=True
)
