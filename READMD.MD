# 🎙 Voice/Text Command Classifier

## 📌 Project Overview
This project builds a **machine learning system** capable of recognizing and classifying both **spoken audio commands** and **written text commands**.  
The goal is to create a unified prediction pipeline that interprets human instructions in multiple formats, enabling natural and flexible interaction with devices.

The system leverages:
- **Text data** (commands stored in CSV files) → processed with **TF‑IDF vectorization**  
- **Audio data** (spoken commands in `.wav` files) → processed with **MFCC feature extraction** using Librosa  
- **Classification models** (Logistic Regression, Random Forest, etc.) → trained to distinguish between command categories  

By combining audio and text features, the classifier can:
- Understand commands whether spoken or typed  
- Provide accurate predictions with robust evaluation metrics (accuracy, confusion matrix, classification report)  
- Support deployment for real‑time inference, enabling applications like smart home control, voice assistants, or accessibility tools  

---

## 📂 Repository Structure
- `app.py` → Streamlit interface for demo and deployment  
- `requirements.txt` → Dependencies (Librosa, scikit‑learn, pandas, etc.)  
- `README.md` → Project documentation  
- `data/` → CSV files for text commands, `.wav` files for audio commands  
- *(Optional)* `notebooks/` → EDA and model experimentation  

---

## 🛠 Steps in the Notebook
1. **Load Data** → Import text and audio datasets  
2. **Preprocess Text** → Tokenization + TF‑IDF vectorization  
3. **Preprocess Audio** → Extract MFCC features with Librosa  
4. **Baseline Models** → Logistic Regression, Random Forest  
5. **Evaluation** → Accuracy, confusion matrix, classification report  
6. **Deployment** → Streamlit app for real‑time inference  

---

## 📊 Results
*(To be updated after EDA)*  
- Dataset distribution analysis  
- Baseline model performance metrics  
- Insights on text vs. audio classification accuracy  

---

## 🔮 Next Steps
- Extend with **CNN for audio spectrograms**  
- Fine‑tune **BERT for text commands**  
- Build a **fusion model** combining both modalities  
- Optimize hyperparameters and evaluate with cross‑validation  
- Add **dual voice input**: microphone capture and audio file upload  

---

## 📚 Resources
- Libraries: pandas, scikit‑learn, librosa, matplotlib, seaborn  
- Dataset: Custom CSV + `.wav` files  

---

## 👩‍💻 Author
**Olorunosebi Dhikrat Motunrayo**
