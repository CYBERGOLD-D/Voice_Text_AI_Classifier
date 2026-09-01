import streamlit as st
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import socket
from sklearn.preprocessing import StandardScaler

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

# --- Text Command Classification ---
st.header("⌨️ Text Command Classification")
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

            # Feature importance visualization (top words)
            if hasattr(text_model, "coef_"):
                importances = text_model.coef_[0]
                indices = np.argsort(importances)[-10:]
                top_words = [vectorizer.get_feature_names_out()[i] for i in indices]
                top_importances = importances[indices]

                fig_imp, ax_imp = plt.subplots()
                ax_imp.barh(top_words, top_importances, color="green")
                ax_imp.set_title("Top Words Driving Prediction")
                st.pyplot(fig_imp)

        # Feedback buttons
        col_fb1, col_fb2 = st.columns(2)
        if col_fb1.button("👍 Text Prediction Correct"):
            st.session_state.feedback.append(("Text", text_command, label, "Positive"))
        if col_fb2.button("👎 Text Prediction Wrong"):
            st.session_state.feedback.append(("Text", text_command, label, "Negative"))

        st.session_state.history.append(("Text", text_command, label, probs.max()))
    else:
        st.warning("Please enter a command.")

# --- Audio Command Classification ---
st.header("📂 Audio Command Classification")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

if audio_file is not None and audio_model:
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
    audio_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # --- Adjust to 7 features + scale ---
    audio_features = StandardScaler().fit_transform(audio_features[:, :7])

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

            # Feature importance visualization (MFCCs)
            st.subheader("Top MFCC Features Driving Prediction")
            fig_mfcc, ax_mfcc = plt.subplots()
            ax_mfcc.bar(range(1, len(audio_features[0]) + 1), audio_features[0], color="purple")
            ax_mfcc.set_xlabel("MFCC Coefficient")
            ax_mfcc.set_ylabel("Value")
            ax_mfcc.set_title("MFCC Feature Contributions")
            st.pyplot(fig_mfcc)

        # Feedback buttons
        col_fb1, col_fb2 = st.columns(2)
        if col_fb1.button("👍 Audio Prediction Correct"):
            st.session_state.feedback.append(("Audio", audio_file.name, label, "Positive"))
        if col_fb2.button("👎 Audio Prediction Wrong"):
            st.session_state.feedback.append(("Audio", audio_file.name, label, "Negative"))

        st.session_state.history.append(("Audio", audio_file.name, label, probs.max()))

# --- Combined History Table ---
st.header("📜 Prediction History")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history, columns=["Type", "Input", "Predicted Label", "Confidence"])

    # --- Summary panel ---
    st.subheader("📊 Summary Insights")
    avg_conf = df_history["Confidence"].mean()
    total_preds = len(df_history)
    pos_feedback = sum(1 for fb in st.session_state.feedback if fb[3] == "Positive")
    total_feedback = len(st.session_state.feedback)
    pos_rate = (pos_feedback / total_feedback * 100) if total_feedback > 0 else 0

    st.markdown(f"""
    - **Total Predictions:** {total_preds}  
    - **Average Confidence:** {avg_conf:.2f}  
    - **Positive Feedback Rate:** {pos_rate:.1f}%
    """)

    # --- Filter option ---
    filter_choice = st.radio("Filter history by type:", ["All", "Text", "Audio"], horizontal=True)
    if filter_choice != "All":
        df_history = df_history[df_history["Type"] == filter_choice]

    # --- Confidence threshold slider ---
    threshold = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.0, 0.05)
    df_history = df_history[df_history["Confidence"] >= threshold]

    st.dataframe(df_history)

    # --- Confidence distribution histogram ---
    st.subheader("Confidence Distribution")
    fig_conf, ax_conf = plt.subplots()
    sns.histplot(df_history["Confidence"], bins=10, kde=True, ax=ax_conf, color="blue")
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Frequency")
    ax_conf.set_title("Distribution of Prediction Confidence")
    st.pyplot(fig_conf)

    csv_history = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History (CSV)", csv_history, "prediction_history.csv", "text/csv")
else:
    st.info("No predictions yet. Try classifying text or audio commands!")

# --- Feedback Table ---
st.header("📝 Feedback Log")
if st.session_state.feedback:
    df_feedback = pd.DataFrame(st.session_state.feedback, columns=["Type", "Input", "Predicted Label", "Feedback"])

    # --- Filter option for feedback ---
    filter_feedback = st.radio("Filter feedback by type:", ["All", "Text", "Audio"], horizontal=True)
    if filter_feedback != "All":
        df_feedback = df_feedback[df_feedback["Type"] == filter_feedback]

    st.dataframe(df_feedback)

    csv_feedback = df_feedback.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Feedback (CSV)", csv_feedback, "prediction_feedback.csv", "text/csv")
else:
    st.info("No feedback yet. Use 👍 or 👎 after predictions.")
