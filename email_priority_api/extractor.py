import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure these are downloaded only once manually or from a setup script
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class EmailFeatureExtractor:
    def __init__(self, max_features=1000, tfidf_vectorizer=None, label_encoder=None):
        self.max_features = max_features
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = tfidf_vectorizer if tfidf_vectorizer else TfidfVectorizer(max_features=max_features)
        self.label_encoder = label_encoder if label_encoder else LabelEncoder()

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = ' '.join(
            self.lemmatizer.lemmatize(token)
            for token in word_tokenize(text)
            if token not in self.stop_words and len(token) > 2
        )
        return text

    def extract_metadata_features(self, df):
        features = pd.DataFrame()
        features['subject_length'] = df['subject'].fillna('').str.len()
        features['subject_word_count'] = df['subject'].fillna('').str.split().str.len()
        features['message_length'] = df['message'].fillna('').str.len()
        features['message_word_count'] = df['message'].fillna('').str.split().str.len()
        features['sender_is_executive'] = df['from'].str.lower().str.contains(
            'ceo|cfo|president|vp|director', na=False).astype(int)
        features['has_date'] = df['date'].notna().astype(int)
        return features

    def extract_features(self, df, is_training=True):
        combined_text = df['subject'].fillna('') + ' ' + df['message'].fillna('')
        processed_text = combined_text.apply(self.preprocess_text)

        if is_training:
            tfidf_features = self.tfidf.fit_transform(processed_text)
        else:
            tfidf_features = self.tfidf.transform(processed_text)

        metadata_features = self.extract_metadata_features(df)

        X = np.hstack([
            tfidf_features.toarray(),
            metadata_features.values
        ])

        y = None
        if 'importance' in df.columns:
            if is_training:
                y = self.label_encoder.fit_transform(df['importance'])
            else:
                y = self.label_encoder.transform(df['importance'])

        feature_names = (
            self.tfidf.get_feature_names_out().tolist() +
            metadata_features.columns.tolist()
        )

        return X, y, feature_names

    @staticmethod
    def parse_email_headers(message):
        headers = {}
        lines = message.split('\n')
        current_key = None
        for line in lines:
            if line.strip() == '':
                break
            if ': ' in line and not line.startswith(' '):
                key, value = line.split(': ', 1)
                current_key = key.lower()
                headers[current_key] = value.strip()
            elif current_key and line.startswith(' '):
                headers[current_key] += ' ' + line.strip()
        return headers

    @staticmethod
    def label_emails(df):
        df['headers'] = df['message'].apply(EmailFeatureExtractor.parse_email_headers)
        df['subject'] = df['headers'].apply(lambda x: x.get('subject', ''))
        df['from'] = df['headers'].apply(lambda x: x.get('from', ''))
        df['date'] = df['headers'].apply(lambda x: x.get('date', ''))

        priority_keywords = ['urgent', 'asap', 'important', 'priority', 'critical']
        exec_titles = ['ceo', 'cfo', 'president', 'vp', 'director']

        conditions = (
            df['subject'].str.lower().apply(lambda x: any(k in x for k in priority_keywords)) |
            df['from'].str.lower().apply(lambda x: any(t in x for t in exec_titles)) |
            df['message'].str.lower().str.contains('urgent|asap|immediate|emergency')
        )

        df['importance'] = 'normal'
        df.loc[conditions, 'importance'] = 'high'
        return df
