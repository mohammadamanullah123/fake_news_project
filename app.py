# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # Initialize app
# app = Flask(__name__)

# # Load model and vectorizer
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Download NLTK resources (if not already done)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Text preprocessing function
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return ' '.join(tokens)

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     user_input = request.form['news_text']
#     cleaned_input = preprocess_text(user_input)
#     input_tfidf = vectorizer.transform([cleaned_input])
#     prediction = model.predict(input_tfidf)[0]
#     confidence = model.predict_proba(input_tfidf)[0]

#     return jsonify({
#         "prediction": prediction,
#         "confidence": f"{max(confidence):.2%}"
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
