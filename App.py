import streamlit as st
import joblib
import numpy as np
import librosa
import sounddevice as sd
import tempfile
import wave

# --- Load saved models and vectorizer ---
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # text preprocessing pipeline

st.title("🎙 Audio & Text Command Classifier")
st.write("Classify commands from text, uploaded audio, or spoken voice using the trained model.")

# --- Text Command Input ---
st.header("Text Command Classification")
text_command = st.text_input("Enter a command (e.g., 'turn on the light')")

if st.button("Classify Text"):
    if text_command.strip() != "":
        text_features = vectorizer.transform([text_command])
        prediction = model.predict(text_features)[0]
        st.success(f"Predicted Class for Text: {prediction}")
    else:
        st.warning("Please enter a command.")

# --- Audio File Input ---
st.header("Audio Command Classification")
audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if audio_file is not None:
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
        audio_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

        if st.button("Classify Uploaded Audio"):
            prediction = model.predict(audio_features)[0]
            st.success(f"Predicted Class for Audio: {prediction}")
    except Exception as e:
        st.error(f"Error processing audio file: {e}")

# --- Spoken Voice Input ---
st.header("Spoken Voice Command Classification")
duration = st.slider("Recording duration (seconds)", 1, 5, 3)

if st.button("Record and Classify Voice"):
    st.info("Recording... Speak now!")
    recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()  # wait until recording is finished

    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        with wave.open(tmpfile.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        voice_path = tmpfile.name

    try:
        y, sr = librosa.load(voice_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
        voice_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

        prediction = model.predict(voice_features)[0]
        st.success(f"Predicted Class for Spoken Voice: {prediction}")
    except Exception as e:
        st.error(f"Error processing recorded voice: {e}")
