from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model artifacts
model = joblib.load('model/spam_classifier.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
encoder = joblib.load('model/label_encoder.joblib')

def clean_text(text):
    text = text.lower()
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'Missing "message" field'}), 400

    cleaned = clean_text(data['message'])
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = encoder.inverse_transform([prediction])[0]

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
