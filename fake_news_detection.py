import pandas as pd
import numpy as np
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import pytesseract

# Set path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

# Function to download NLTK data (runs only once)
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        st.stop()

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to load or train model and vectorizer
def load_or_train_model():
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        st.write("✅ Loaded saved model and vectorizer.")
    else:
        dataset_paths = ['news.csv', '../archive/news.csv']
        df = None
        for path in dataset_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"📄 Dataset loaded successfully from {path}.")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.error(f"Error loading dataset from {path}: {e}")
                continue

        if df is None:
            st.error("❌ Dataset 'news.csv' not found in any specified location. Please place it in the project folder or adjust the path.")
            st.stop()

        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("Dataset must have 'text' and 'label' columns.")
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

        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"📊 Model Accuracy on Test Set: `{accuracy:.2%}`")

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        st.success("✅ Trained and saved model and vectorizer.")

    return model, vectorizer

# === Main Streamlit App ===

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown('<div class="title">📰 Fake News Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a news article below to check if it\'s <strong>FAKE</strong> or <strong>REAL</strong>.</div>', unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            padding: 2rem;
        }
        .stTextArea, .stButton {
            margin-top: 1rem;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #574fd6;
        }
        .confidence {
            font-weight: bold;
            color: #444;
        }
        .title {
            color: #3f51b5;
            font-size: 2.2em;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.1em;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# NLTK download
download_nltk_data()

# Load model/vectorizer
model, vectorizer = load_or_train_model()

# Manual Text Input
user_input = st.text_area("✍️ Enter the news article text:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip():
        cleaned_input = preprocess_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_tfidf)[0]
        probability = model.predict_proba(input_tfidf)[0]

        confidence = probability[1] if prediction == 'REAL' else probability[0]
        confidence_percent = f"{confidence:.2%}"

        if prediction == 'REAL':
            st.success(f"✅ This news is likely REAL.")
            st.markdown(f"<p class='confidence'>📈 Confidence: {confidence_percent}</p>", unsafe_allow_html=True)
        else:
            st.error(f"⚠️ This news is likely FAKE.")
            st.markdown(f"<p class='confidence'>📉 Confidence: {confidence_percent}</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text before clicking Predict.")

# Image Upload and Prediction
st.write("📷 Or upload a news article image:")

uploaded_image = st.file_uploader("Upload News Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼️ Uploaded News Image", use_container_width=True)

    with st.spinner("🔍 Extracting text from image..."):
        extracted_text = pytesseract.image_to_string(image)

    st.success("✅ Text successfully extracted from image:")
    st.text_area("📝 Extracted Text", value=extracted_text, height=150)

    if st.button("📢 Predict from Image"):
        cleaned_input = preprocess_text(extracted_text)
        input_tfidf = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_tfidf)[0]
        probability = model.predict_proba(input_tfidf)[0]

        confidence = probability[1] if prediction == 'REAL' else probability[0]
        confidence_percent = f"{confidence:.2%}"

        if prediction == 'REAL':
            st.success(f"✅ This news is likely REAL.")
            st.markdown(f"<p class='confidence'>📈 Confidence: {confidence_percent}</p>", unsafe_allow_html=True)
        else:
            st.error(f"⚠️ This news is likely FAKE.")
            st.markdown(f"<p class='confidence'>📉 Confidence: {confidence_percent}</p>", unsafe_allow_html=True)
