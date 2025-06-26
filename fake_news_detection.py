import pandas as pd
import numpy as np
import nltk
import joblib
import os
import re
from PIL import Image
import pytesseract
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 🛠 Tesseract Path Fix
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("❌ Tesseract not found! Please install it from https://github.com/tesseract-ocr/tesseract and adjust the path.")
    st.stop()

# 📥 NLTK Data
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# 🧼 Text Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 🤖 Model Loader/Trainer
def load_or_train_model():
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        df = None
        for path in ['news.csv', '../archive/news.csv']:
            try:
                df = pd.read_csv(path)
                st.success(f"📄 Loaded dataset from {path}")
                break
            except:
                continue

        if df is None or 'text' not in df.columns or 'label' not in df.columns:
            st.error("❌ Dataset must include 'text' and 'label' columns.")
            st.stop()

        df['cleaned_text'] = df['text'].apply(preprocess_text)
        X = df['cleaned_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)

        acc = accuracy_score(y_test, model.predict(X_test_tfidf))
        st.write(f"📊 Accuracy: {acc:.2%}")

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer

# 🌐 Streamlit UI Setup
st.set_page_config(page_title="Fake News Detector", layout="wide")

# 💄 Custom UI Styling
st.markdown("""<style>
[data-testid="stAppViewContainer"] {
 background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
 background-size: 400% 400%;
 animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.main {
 background-color: rgba(255, 255, 255, 0.85);
 backdrop-filter: blur(10px);
 border-radius: 20px;
 padding: 2rem;
 box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
 margin: 2rem auto;
 max-width: 1000px;
}
.card {
 background: rgba(255, 255, 255, 0.7);
 backdrop-filter: blur(5px);
 border-radius: 15px;
 padding: 1.5rem;
 box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
 margin-bottom: 1.5rem;
 border: 1px solid rgba(255, 255, 255, 0.2);
}
.title {
 text-align: center;
 font-size: 2.8rem;
 color: #2c3e50;
 margin-bottom: 0.5rem;
 font-weight: 700;
 text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.subtitle {
 text-align: center;
 font-size: 1.3rem;
 color: #2c3e50;
 margin-bottom: 2rem;
 font-weight: 400;
}
.stTextArea textarea {
 background: rgba(255, 255, 255, 0.8) !important;
 border: 2px solid rgba(0, 0, 0, 0.1) !important;
 border-radius: 12px !important;
 padding: 1rem !important;
}
.stTextArea textarea:focus {
 border-color: #6e48aa !important;
 box-shadow: 0 0 0 3px rgba(110, 72, 170, 0.1) !important;
 background: white !important;
}
.stButton>button {
 background: linear-gradient(to right, #6e48aa, #9d50bb);
 color: white;
 border: none;
 border-radius: 12px;
 padding: 12px 24px;
 font-size: 1rem;
 font-weight: 500;
 transition: all 0.3s ease;
 box-shadow: 0 4px 15px rgba(110, 72, 170, 0.4);
}
.stButton>button:hover {
 transform: translateY(-2px);
 box-shadow: 0 6px 20px rgba(110, 72, 170, 0.6);
}
.stButton>button:active {
 transform: translateY(0);
}
.real-result {
 background: rgba(79, 172, 254, 0.15);
 border: 1px solid rgba(79, 172, 254, 0.3);
 color: #2c3e50;
}
.fake-result {
 background: rgba(245, 87, 108, 0.15);
 border: 1px solid rgba(245, 87, 108, 0.3);
 color: #2c3e50;
}
.confidence {
 font-size: 1.1rem;
 font-weight: 600;
 color: white;
 background: linear-gradient(135deg, #6e48aa, #9d50bb);
 padding: 8px 16px;
 border-radius: 50px;
 display: inline-block;
 box-shadow: 0 4px 15px rgba(110, 72, 170, 0.3);
}
@keyframes fadeIn {
 from { opacity: 0; transform: translateY(10px); }
 to { opacity: 1; transform: translateY(0); }
}
.stMarkdown, .stTextArea, .stFileUploader {
 animation: fadeIn 0.5s ease-out;
}
</style>""", unsafe_allow_html=True)

# 📰 App Title
st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check whether a news article is <strong>REAL</strong> or <strong>FAKE</strong> using AI</div>', unsafe_allow_html=True)

# 🚀 Start
download_nltk_data()
model, vectorizer = load_or_train_model()

col1, col2 = st.columns(2)

# ✍️ Text Input
with col1:
    st.subheader("📝 Enter News Text")
    user_input = st.text_area("Paste or type news article:", height=200)
    if st.button("🔍 Predict Text"):
        if user_input.strip():
            cleaned = preprocess_text(user_input)
            tfidf = vectorizer.transform([cleaned])
            prediction = model.predict(tfidf)[0]
            prob = model.predict_proba(tfidf)[0]
            conf = prob[1] if prediction == 'REAL' else prob[0]
            if prediction == 'REAL':
                st.success("✅ This news is likely REAL")
            else:
                st.error("⚠️ This news is likely FAKE")
            st.markdown(f"<div class='confidence'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text")

# 🖼️ Image Upload & OCR
with col2:
    st.subheader("📷 Upload News Image")
    uploaded_image = st.file_uploader("Upload image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        with st.spinner("Extracting text..."):
            extracted = pytesseract.image_to_string(image, lang='eng+hin+urd+pan')
        st.text_area("Extracted Text:", value=extracted, height=150)
        if st.button("📢 Predict from Image"):
            cleaned = preprocess_text(extracted)
            tfidf = vectorizer.transform([cleaned])
            prediction = model.predict(tfidf)[0]
            prob = model.predict_proba(tfidf)[0]
            conf = prob[1] if prediction == 'REAL' else prob[0]
            if prediction == 'REAL':
                st.success("✅ This news is likely REAL")
            else:
                st.error("⚠️ This news is likely FAKE")
            st.markdown(f"<div class='confidence'>Confidence: {conf:.2%}</div>", unsafe_allow_html=True)
