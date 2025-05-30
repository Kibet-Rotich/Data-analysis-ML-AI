from flask import Flask, request
from flask_restx import Api, Resource, fields
import pickle
import pandas as pd
from extractor import EmailFeatureExtractor

app = Flask(__name__)
api = Api(app, title="Enron Email Priority API", version="1.0", description="Classifies Enron emails as 'high' or 'normal' priority")

# Namespace for clarity
ns = api.namespace('priority', description='Email Priority Classification')

# Swagger input model
message_model = ns.model('Message', {
    'message': fields.String(required=True, description='Raw email text (with headers and body)')
})

# Load saved model and tools
with open('model/rf_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

extractor = EmailFeatureExtractor(tfidf_vectorizer=tfidf_vectorizer, label_encoder=label_encoder)

@ns.route('/predict')
class PriorityClassifier(Resource):
    @ns.expect(message_model)
    @ns.doc(description="Predict the priority of an email (high or normal).")
    def post(self):
        data = request.json
        if 'message' not in data:
            return {'error': 'Missing "message" field'}, 400

        msg = data['message']
        df = pd.DataFrame({'message': [msg], 'file': ['user_input']})

        df = extractor.label_emails(df)
        X, _, _ = extractor.extract_features(df, is_training=False)
        prediction = classifier.predict(X)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        return {'priority': label}, 200

if __name__ == '__main__':
    app.run(debug=True)
