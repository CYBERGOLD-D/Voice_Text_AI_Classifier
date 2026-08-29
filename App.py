import streamlit as st
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Safe model loading ---
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Could not load {name}: {e}")
        return None

# --- Load models and vectorizer ---
audio_model = safe_load("best_model.pkl", "Audio Model")  # Logistic Regression (Audio)
text_model = safe_load("text_model.pkl", "Text Model")    # Random Forest (Text)
vectorizer = safe_load("text_tfidf_vectorizer.pkl", "Text Vectorizer")

# --- Optional: class label mapping ---
class_labels = {
    0: "Turn on Light",
    1: "Turn off Light",
    2: "Play Music",
    3: "Stop Music",
    # ... continue mapping up to 20
}

st.title("🎙 Voice & Text Command Classifier")

# --- Quick Start Guide ---
st.markdown("""
## 🚀 Quick Start Guide
Welcome! This app lets you classify commands using text or audio.

**How to use:**
1. ✍️ Type a command in the text box or pick a sample.
2. 🎵 Upload an audio file (wav, mp3, ogg, flac, m4a).
3. 📊 See the prediction, confidence level, and your history.
4. ⬇️ Download your history as a CSV file.

**Tips:**
- Try short, clear commands for best results.
- You can upload recordings from any phone or device.
- Use the feedback buttons to tell us if the prediction was correct.
""")

# --- Session history ---
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# --- Helper functions ---
def predict_text(command):
    features = vectorizer.transform([command])
    pred = text_model.predict(features)[0]
    prob = text_model.predict_proba(features).max()
    return pred, prob

def predict_audio_features(features):
    pred = audio_model.predict(features)[0]
    prob = audio_model.predict_proba(features).max()
    return pred, prob

# --- Text Command Classification ---
st.header("⌨️ Text Command Classification")
sample = st.selectbox("Or try a sample command:", ["Turn on Light", "Turn off Light", "Play Music", "Stop Music"])
text_command = st.text_input("Enter a command", value=sample)

if st.button("Classify Text") and text_model and vectorizer:
    if text_command.strip():
        pred, prob = predict_text(text_command)
        label = class_labels.get(pred, pred)
        st.success(f"Predicted Class: {label} (Confidence: {prob:.2f})")
        st.progress(int(prob * 100))
        st.session_state.history.append(("Text", text_command, label, prob))
        if st.button("👍 Correct"):
            st.session_state.feedback.append(("Text", text_command, "Correct"))
        if st.button("👎 Incorrect"):
            st.session_state.feedback.append(("Text", text_command, "Incorrect"))
    else:
        st.warning("Please enter a command.")

# --- Audio File Input ---
st.header("📂 Audio Command Classification")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

if audio_file is not None and audio_model:
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)  # match training setup
        audio_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

        # Show waveform
        st.subheader("🔊 Audio Waveform")
        fig_wave, ax_wave = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        ax_wave.set_title("Waveform")
        st.pyplot(fig_wave)

        # Show spectrogram
        st.subheader("🌈 Spectrogram")
        fig_spec, ax_spec = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax_spec)
        fig_spec.colorbar(img, ax=ax_spec, format="%+2.f dB")
        ax_spec.set_title("Mel Spectrogram")
        st.pyplot(fig_spec)

        if st.button("Classify Uploaded Audio"):
            pred, prob = predict_audio_features(audio_features)
            label = class_labels.get(pred, pred)
            st.success(f"Predicted Class: {label} (Confidence: {prob:.2f})")
            st.progress(int(prob * 100))
            st.session_state.history.append(("Audio File", audio_file.name, label, prob))
            if st.button("👍 Correct"):
                st.session_state.feedback.append(("Audio", audio_file.name, "Correct"))
            if st.button("👎 Incorrect"):
                st.session_state.feedback.append(("Audio", audio_file.name, "Incorrect"))
    except Exception as e:
        st.error(f"Error processing audio file: {e}")

# --- History Log ---
st.header("📝 Command History")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history, columns=["Type", "Input", "Prediction", "Confidence"])
    st.dataframe(df_history)

    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History as CSV", csv, "command_history.csv", "text/csv")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared successfully!")
else:
    st.write("No commands classified yet.")

# --- Feedback Log ---
st.header("🗣️ User Feedback")
if st.session_state.feedback:
    df_feedback = pd.DataFrame(st.session_state.feedback, columns=["Type", "Input", "Feedback"])
    st.dataframe(df_feedback)

# --- Model Info ---
st.header("ℹ️ Model Information")
st.write("**Text Model:** Random Forest Classifier")
st.write("- n_estimators=300, max_depth=50, class_weight='balanced'")
st.write("- Trained on TF-IDF features (max_features=5000)")
st.write("**Audio Model:** Logistic Regression Classifier")
st.write("- C=0.1, solver='saga', class_weight='balanced', max_iter=1000")
st.write("- Trained on MFCC features (7 coefficients, averaged)")
