# 🎙 Voice & Text Command Classifier  
*Smart multimodal AI for text and audio commands*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Prototype-yellow.svg)

---

## 📌 Overview
This project is a **multimodal command classifier** that recognizes and classifies user commands from **text, uploaded audio files, or spoken voice recordings**.  

It uses **Random Forest models** trained on:
- **Text features**: TF‑IDF vectors (max_features=5000)  
- **Audio features**: MFCCs (14 coefficients, averaged)

The app is built with **Streamlit** for an interactive interface and supports smart‑home style actions (e.g., toggling lights, playing music).

---

## 🚀 Features
- ⌨️ Text Classification  
- 📂 Audio File Classification  
- 🎤 Spoken Voice Classification  
- 📈 Confidence Scores  
- 📝 Command History with CSV export  
- 🗑️ Clear History option  
- 📊 Performance Dashboard  
- 📉 Live Confusion Matrix  
- ℹ️ Model Info Section  

---

## 🛠️ Installation
```bash
git clone https://github.com/CYBERGOLD-D/Voice_Text_AI_Classifier.git
cd Voice_Text_AI_Classifier
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run App.py
```

---

## 📂 Project Structure
```
Voice_Text_AI_Classifier/
│── App.py                     # Streamlit app
│── Audio_Text_CC_Full.ipynb   # Training & evaluation notebook
│── datasets/                  # Raw and processed datasets
│── text_model.pkl             # Saved Random Forest text model
│── best_model.pkl             # Saved Random Forest audio model
│── text_tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
│── requirements.txt           # Dependencies
│── README.md                  # Documentation
```

---

## ✅ Current Baseline
- **Best Fit Model**: Random Forest (Text features)  
- **Accuracy**: ~5% (baseline, close to random guessing across 21 classes)  
- **Limitations**: Class imbalance and limited feature representation  

---

## 🔮 Future Work
- Advanced audio features & transformer embeddings  
- Non‑linear models (XGBoost, LightGBM, deep learning)  
- Multimodal fusion (combine audio + text features)  
- Stronger evaluation with k‑fold cross‑validation  

---

## 📸 Usage Examples
- **Text Input**: `"turn on the light"` → `Turn on Light (Confidence: 0.82)`  
- **Audio Upload**: `light_command.wav` → `Turn on Light (Confidence: 0.76)`  
- **Voice Recording**: `"play music"` → `Play Music (Confidence: 0.80)`  

---

## 👩‍💻 Author
**Olorunosebi Dhikrat Motunrayo**  
📍 Abuja, Nigeria  

---

## 🧠 About
**Voice & Text Command Classifier** is a smart multimodal AI system that understands both **spoken voice** and **written text** commands. Built with **Streamlit** and powered by **Random Forest models**, it classifies user inputs from audio or text into actionable categories — enabling intuitive, voice‑driven and text‑based control for smart environments.  

Developed by **Olorunosebi Dhikrat Motunrayo**, this project blends **machine learning**, **signal processing**, and **natural language understanding** into one seamless interactive experience.

---

## 🧾 Tagline
> “Empowering machines to understand both voice and text — one command at a time.”
```

---

