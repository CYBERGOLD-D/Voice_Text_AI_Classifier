import streamlit as st
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import socket

# --- Safe model loading ---
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Could not load {name}: {e}")
        return None

# --- Load models and vectorizer ---
audio_model = safe_load("best_model.pkl", "Audio Model")
text_model = safe_load("text_model.pkl", "Text Model")
vectorizer = safe_load("text_tfidf_vectorizer.pkl", "Text Vectorizer")

# --- Class label mapping ---
class_labels = {
    0: "Turn on Light",
    1: "Turn off Light",
    2: "Play Music",
    3: "Stop Music",
    # ... extend up to 20
}

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# --- App Layout ---
st.title("🎙 Voice & Text Command Classifier")

st.markdown("""
## 🚀 Quick Start Guide
1. ✍️ Type a command or pick a sample.
2. 🎵 Upload an audio file (wav, mp3, ogg, flac, m4a).
3. 📊 See prediction + confidence bar.
4. 👍👎 Give feedback.
5. ⬇️ Download your history as CSV.
""")

# --- Toggle for advanced visualizations ---
show_advanced = st.checkbox("Show advanced visualizations (probability charts & feature importance)", value=True)

# --- Side-by-side layout for Text and Audio ---
st.header("🔀 Text & Audio Classification Side-by-Side")
col1, col2 = st.columns(2)

with col1:
    st.subheader("⌨️ Text Command Classification")
    sample = st.selectbox("Try a sample:", ["Turn on Light", "Turn off Light", "Play Music", "Stop Music"])
    text_command = st.text_input("Enter a command", value=sample)

    if st.button("Classify Text") and text_model and vectorizer:
        if text_command.strip():
            features = vectorizer.transform([text_command])
            pred = text_model.predict(features)[0]
            probs = text_model.predict_proba(features)[0]
            label = class_labels.get(pred, pred)

            st.success(f"Predicted Class: {label}")
            st.progress(int(probs.max() * 100))

            if show_advanced:
                # Probability distribution chart
                fig, ax = plt.subplots()
                ax.bar(range(len(probs)), probs, color="skyblue")
                ax.set_xticks(range(len(probs)))
                ax.set_xticklabels([class_labels.get(i, str(i)) for i in range(len(probs))], rotation=45, ha="right")
                ax.set_ylabel("Probability")
                ax.set_title("Probability Distribution")
                st.pyplot(fig)

                # Download probability distribution
                df_probs = pd.DataFrame([probs], columns=[class_labels.get(i, str(i)) for i in range(len(probs))])
                csv_probs = df_probs.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Probability Distribution (CSV)", csv_probs, "text_probabilities.csv", "text/csv")

                # Feature importance visualization (top words)
                if hasattr(text_model, "feature_importances_"):
                    importances = text_model.feature_importances_
                    indices = np.argsort(importances)[-10:]  # top 10 words
                    top_words = [vectorizer.get_feature_names_out()[i] for i in indices]
                    top_importances = importances[indices]

                    fig_imp, ax_imp = plt.subplots()
                    ax_imp.barh(top_words, top_importances, color="green")
                    ax_imp.set_title("Top Words Driving Prediction")
                    st.pyplot(fig_imp)

            st.session_state.history.append(("Text", text_command, label, probs.max()))
        else:
            st.warning("Please enter a command.")

with col2:
    st.subheader("📂 Audio Command Classification")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

    if audio_file is not None and audio_model:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
        audio_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

        if st.button("🎯 Classify Audio"):
            pred = audio_model.predict(audio_features)[0]
            probs = audio_model.predict_proba(audio_features)[0]
            label = class_labels.get(pred, pred)

            st.success(f"Predicted Class: {label}")
            st.progress(int(probs.max() * 100))

            if show_advanced:
                # Probability distribution chart
                fig, ax = plt.subplots()
                ax.bar(range(len(probs)), probs, color="lightcoral")
                ax.set_xticks(range(len(probs)))
                ax.set_xticklabels([class_labels.get(i, str(i)) for i in range(len(probs))], rotation=45, ha="right")
                ax.set_ylabel("Probability")
                ax.set_title("Probability Distribution")
                st.pyplot(fig)

                # Download probability distribution
                df_probs = pd.DataFrame([probs], columns=[class_labels.get(i, str(i)) for i in range(len(probs))])
                csv_probs = df_probs.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Probability Distribution (CSV)", csv_probs, "audio_probabilities.csv", "text/csv")

                # Feature importance visualization (MFCCs)
                st.subheader("Top MFCC Features Driving Prediction")
                fig_mfcc, ax_mfcc = plt.subplots()
                ax_mfcc.bar(range(1, len(audio_features[0]) + 1), audio_features[0], color="purple")
                ax_mfcc.set_xlabel("MFCC Coefficient")
                ax_mfcc.set_ylabel("Value")
                ax_mfcc.set_title("MFCC Feature Contributions")
                st.pyplot(fig_mfcc)

            st.session_state.history.append(("Audio", audio_file.name, label, probs.max()))
