from flask import Flask, request, jsonify
import pickle
import pandas as pd
from extractor import EmailFeatureExtractor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and tools
with open('model/rf_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize extractor with pre-fitted vectorizer and encoder
extractor = EmailFeatureExtractor(tfidf_vectorizer=tfidf_vectorizer, label_encoder=label_encoder)

@app.route('/predict', methods=['POST'])
def predict_priority():
    data = request.json

    if 'message' not in data:
        return jsonify({'error': 'Missing "message" field'}), 400

    # Wrap the message in a DataFrame
    msg = data['message']
    df = pd.DataFrame({'message': [msg], 'file': ['user_input']})

    # Label for consistency (even though label isn't used during inference)
    df = extractor.label_emails(df)

    # Extract features and predict
    X, _, _ = extractor.extract_features(df, is_training=False)
    prediction = classifier.predict(X)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    return jsonify({'priority': label})

if __name__ == '__main__':
    app.run(debug=True)
