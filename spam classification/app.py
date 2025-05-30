from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib

app = Flask(__name__)
api = Api(app, title="Spam Email Classifier API", version="1.0", description="Classifies SMS/email messages as spam or ham")

# Namespace
ns = api.namespace('spam', description='Spam Classification Operations')

# Input model (Swagger docs)
message_model = api.model('Message', {
    'message': fields.String(required=True, description='Email or SMS text')
})

# Load components
model = joblib.load('model/spam_classifier.joblib')
vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
encoder = joblib.load('model/label_encoder.joblib')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    return ''.join([c for c in text if c.isalnum() or c.isspace()])

# Endpoint
@ns.route('/predict')
class SpamClassifier(Resource):
    @ns.doc('predict_spam')
    @ns.expect(message_model)
    def post(self):
        """Classify message as spam or ham"""
        data = request.json
        text = clean_text(data['message'])
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        label = encoder.inverse_transform([prediction])[0]
        return {'label': label}, 200

if __name__ == '__main__':
    app.run(debug=True)
