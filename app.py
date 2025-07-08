from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import os
import re
import platform
import pandas as pd
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# ✅ Tesseract Path for Windows only
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))


# 🌐 Flask App Setup
app = Flask(__name__)
app.secret_key = 'secret123'

# 🧼 Text Preprocessing Function
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

# 🤖 Model Loader or Trainer
def load_or_train_model():
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("✅ Model files found. Loading...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("⚠️ Model files not found. Training new model...")
        df = None
        for path in ['news.csv', '../archive/news.csv']:
            try:
                df = pd.read_csv(path)
                break
            except:
                continue

        if df is None or 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must include 'text' and 'label' columns.")

        df['cleaned_text'] = df['text'].apply(preprocess_text)
        X = df['cleaned_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)

        acc = accuracy_score(y_test, model.predict(X_test_tfidf))
        print(f"📊 Accuracy: {acc:.2%}")

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print("✅ Model & vectorizer saved.")

    return model, vectorizer

# ✅ Load model
model, vectorizer = load_or_train_model()

# 🔗 Routes
@app.route('/favicon.png')
def favicon():
    return send_file('favicon.png', mimetype='image/png')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prediction = None
        confidence = None
        extracted_text = None

        if 'text' in request.form and request.form['text'].strip():
            user_text = request.form['text']
            cleaned = preprocess_text(user_text)
            tfidf = vectorizer.transform([cleaned])
            pred = model.predict(tfidf)[0]
            prob = model.predict_proba(tfidf)[0]
            confidence = prob[1] if pred == 'REAL' else prob[0]
            prediction = pred

        elif 'image' in request.files:
            image_file = request.files['image']
            if image_file:
                image = Image.open(image_file)
                extracted_text = pytesseract.image_to_string(image, lang='eng')
                cleaned = preprocess_text(extracted_text)
                tfidf = vectorizer.transform([cleaned])
                pred = model.predict(tfidf)[0]
                prob = model.predict_proba(tfidf)[0]
                confidence = prob[1] if pred == 'REAL' else prob[0]
                prediction = pred

        session['prediction'] = f"✅ This news is likely REAL" if prediction == 'REAL' else "❌ This news is likely FAKE"
        session['confidence'] = round(confidence * 100, 2) if confidence else None
        session['extracted'] = extracted_text
        return redirect(url_for('index'))

    return render_template("index.html",
        prediction=session.pop('prediction', None),
        confidence=session.pop('confidence', None),
        extracted=session.pop('extracted', None)
    )

# 🚀 Run locally
if __name__ == '__main__':
    app.run(debug=True)
