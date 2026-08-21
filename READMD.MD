🎙 Voice/Text Command Classifier
📌 Project Overview
This project develops a machine learning system that recognizes and classifies both spoken audio commands and written text commands.
The aim is to build a unified pipeline that interprets human instructions across modalities, enabling natural and flexible device interaction.

The system leverages:

Text data (CSV commands) → processed with TF‑IDF vectorization

Audio data (.wav commands) → processed with MFCC feature extraction via Librosa

Classification models (Logistic Regression, Random Forest, etc.) → trained to distinguish command categories

By combining audio and text features, the classifier can:

Interpret commands whether spoken or typed

Deliver predictions with evaluation metrics (accuracy, confusion matrix, classification report)

Support deployment for real‑time inference (smart home control, voice assistants, accessibility tools)

📂 Repository Structure
(to be expanded with folders and files)

🛠 Steps in the Notebook
Load Data → Import text and audio datasets

Preprocess Text → Tokenization + TF‑IDF vectorization

Preprocess Audio → Extract MFCC features with Librosa

Train Models → Logistic Regression, Random Forest

Evaluate Models → Accuracy, confusion matrix, classification report

Deploy → Streamlit app for real‑time inference

📊 Training & Evaluation Results
Dataset
Training set: (38359, 14) features and (38359, 21) labels

Testing set: (9590, 14) features and (9590, 21) labels

Model Training
✅ Logistic Regression trained successfully

Samples: 38,359

Features: 14

✅ Random Forest trained successfully

Samples: 38,359

Features: 14

Model Performance
Logistic Regression

Accuracy: 0.0461

Classification report shows low precision/recall across categories

Random Forest

Accuracy: 0.0418

Similar performance, slightly lower than Logistic Regression

Summary
🎉 All models trained successfully

✅ Best model: Logistic Regression (saved as best_model.pkl)

✅ Vectorizer retrained and saved (tfidf_vectorizer.pkl)

🔮 Next Steps
Extend with CNNs for audio spectrograms

Fine‑tune BERT for text commands

Build a fusion model combining both modalities

Optimize hyperparameters with cross‑validation

Add dual voice input: microphone capture + audio file upload

🛠 Installation (Local)
Clone the repository:


git clone https://github.com/CYBERGODLD-D/Voice_Text_AI_Classifier.git
cd Voice_Text_AI_Classifier
Install dependencies:


pip install -r requirements.txt
Run the Streamlit app:


streamlit run app.py
👩‍💻 Author  
Olorunosebi Dhikrat Motunrayo