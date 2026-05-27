<h1 align=\"center\">🧠 AI Sentiment Pro</h1>

<p align=\"center\">
  <strong>A Complete, Production-Ready Sentiment Analysis & Emotion Detection System</strong>
  <strong>I have Attached an .mp4 file,do visit and check the result named as AI-Sentiment-pro.mp4</strong>
</p>
<hr>

## 📖 Overview

**AI Sentiment Pro** is an advanced Natural Language Processing (NLP) web application built using Python, Scikit-Learn, and Streamlit. It takes text or voice input and predicts its underlying sentiment (Positive, Negative, Neutral) in real-time. 

Beyond standard sentiment classification, this system incorporates **Emotion Detection**, **Multilingual Support**, and **Fake Review Flagging**, making it a robust, enterprise-grade prototype for customer feedback analysis.

---


## ✨ Key Features

- ⚡ **Real-Time Sentiment Analysis:** Instantly analyze text via an interactive UI using custom-trained ML models (Logistic Regression, Naive Bayes, SVM).
- 🎭 **Emotion Detection:** Heuristic-based detection mapping sentiment and keywords to specific emotions (Joy, Sadness, Anger, Fear, Surprise).
- 🎙️ **Voice-to-Text Capabilities:** Speak directly into your microphone, and the system will transcribe and analyze your audio live.
- 🌍 **Multilingual Support:** Powered by `deep-translator`, non-English text is automatically translated to English for highly accurate ML analysis.
- 🚫 **Fake Review Detection:** Flags potentially spammy, bot-generated, or artificially inflated text based on punctuation patterns and keywords.
- 📊 **Batch Processing:** Upload a CSV file of reviews for bulk sentiment predictions, returning downloadable results alongside interactive Plotly charts and Word Clouds.
- 📈 **Model Analytics Dashboard:** View model accuracy, confusion matrices, and classification reports directly within the application.

---

## 🏗️ Architecture & Directory Structure

```text
📦 AI-Sentimental-Analysis
 ┣ 📂 data                   # Contains CSV datasets (e.g., sample_dataset.csv)
 ┣ 📂 models                 # Pickled ML models and TF-IDF vectorizers
 ┣ 📂 utils                  # Helper scripts and modular functions
 ┃ ┣ 📜 advanced.py          # Emotion, Voice, and Multilingual logic
 ┃ ┣ 📜 preprocessing.py     # NLTK text cleaning (Lemmatization, Tokenization)
 ┃ ┗ 📜 visualization.py     # Plotly & Matplotlib charting functions
 ┣ 📜 .gitignore             # Standard Python gitignore
 ┣ 📜 app.py                 # Main Streamlit web application
 ┣ 📜 requirements.txt       # Project dependencies
 ┣ 📜 train.py               # ML training pipeline script
 ┗ 📜 README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Sentimental-Analysis.git
cd AI-Sentimental-Analysis
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Upon first run, the NLTK library will automatically download required corpora like `punkt` and `stopwords` in the background).*

### 4. Train the Machine Learning Models
Generate the dataset, vectorize the text, train the models, and export `.pkl` files to the `/models` directory:
```bash
python train.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---


## 💡 Future Enhancements

- [ ] **Deep Learning Integration:** Upgrade the backend from Scikit-Learn to a Transformer model (like BERT or RoBERTa) using PyTorch or HuggingFace.
- [ ] **Dedicated Emotion Dataset:** Train a secondary Deep Learning model specifically for emotion classification rather than using heuristics.
- [ ] **Database Integration:** Connect to PostgreSQL or MongoDB to save user prediction histories and build a persistent analytics dashboard.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<p align=\"center\">
  <i>Developed with ❤️ for the AI community.</i>
</p>
