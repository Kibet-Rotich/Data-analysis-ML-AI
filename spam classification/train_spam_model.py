import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

# Load raw SMS Spam Collection data
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

# Clean the text
def clean_text(text):
    text = text.lower()
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

df['cleaned_text'] = df['text'].apply(clean_text)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])

# Encode labels ('ham'=0, 'spam'=1)
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save everything
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/spam_classifier.joblib')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')
joblib.dump(le, 'model/label_encoder.joblib')

print("✅ Training complete. Model, vectorizer, and encoder saved.")
