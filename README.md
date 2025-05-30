# 📦 Data Science, AI, and ML Projects

Welcome! I'm **Rotich Kibet**, a passionate **data scientist** and **machine learning engineer**. This repository showcases a variety of end-to-end projects across natural language processing (NLP), machine learning (ML), and real-world data analysis — all developed with a focus on clean pipelines, clarity, and reproducibility.

---

## 🔌 Deployed Machine Learning APIs

### 📧 Enron Email Priority Classification
A machine learning pipeline trained on the Enron Email dataset to classify corporate emails as **"high"** or **"normal"** priority using content, metadata, and keyword signals.

- RESTful API built with Flask and documented with Swagger (Flask-RESTX)
- Feature extraction includes subject line parsing, sender metadata, and content vectors
- Model testing and exploration available in the accompanying Jupyter notebook

📂 Folder: `email_priority_api/`  
📓 Notebook: `Enron_Email_dataset.ipynb`

---

### 🚨 Spam Email Detection
Detects spam messages using the SMS Spam Collection dataset. Includes data preprocessing, TF-IDF vectorization, and a Random Forest model deployed via a Flask API.

- Swagger UI for testing and client consumption
- Model serialization and consistent inference using joblib
- API mirrors model tested in the associated notebook

📂 Folder: `spam classification/`  
📓 Notebook: `Email_Priority_classifier.ipynb`

---

## 📊 Data Analysis Projects

### 🌍 Kenya Economic and Social Indicators
In-depth exploration of Kenya’s development trends using World Bank data.

- Time-series visualizations and correlation analysis
- Focused on GDP, life expectancy, population, and economic indicators
- Cleaned and reshaped datasets for interactive analysis

📓 File: `Kenya.ipynb`

---

### 🌐 World Bank Global Data Analysis
Data cleaning, filtering, and global-level analytics using World Bank datasets.

- Visualizes trends in global military expenditure
- Includes data wrangling techniques and aggregate analysis

📓 File: `Worldbank_data.ipynb`

---

## 🤖 Machine Learning Projects

### 📨 Enron Email Classifier
A machine learning model for classifying Enron emails by importance.

- Advanced feature engineering using email headers, sender tags, and message content
- Built to distinguish high-importance internal communication from general email flow

📓 File: `Enron_Email_dataset.ipynb`

---

### 📨 Email Priority Classifier (SMS-Based)
Adapted from the SMS Spam dataset to classify messages as **"Urgent"** or **"Regular"**.

- Covers full ML pipeline: text cleaning, vectorization, training, and prediction
- Includes notebook-based model evaluation and examples

📓 File: `Email_Priority_classifier.ipynb`

---

## 🚢 Exploratory & Foundational Projects

### 🚢 Titanic Survival Analysis
A classic ML problem revisited with modern techniques.

- Exploratory Data Analysis (EDA), visualization, and logistic regression
- Preprocessing steps include handling missing data and encoding

📓 File: `Titanic.ipynb`

---

## 🧪 Neural Network Sandbox
A space for building and experimenting with neural networks from scratch.

- Testing activation functions, backpropagation, and weight updates
- Good ground for deeper DL understanding and experimentation

📂 Folder: `Neural Networks/`

---

## 🚀 What's Coming

Stay tuned for upcoming projects in:
- **Computer Vision**: Object detection and image classification  
- **Advanced NLP**: Summarization, topic modeling, sentiment classification  
- **MLOps**: Model monitoring, CI/CD for ML pipelines

---

## 📖 How to Use

### Requirements
- **Python** 3.8+
- Core packages: `pandas`, `numpy`, `scikit-learn`, `nltk`, `flask`, `flask-restx`, `joblib`, `matplotlib`, `seaborn`

### Setup Instructions
```bash
git clone https://github.com/Kibet-Rotich/Data-analysis-ML-AI.git
cd Data-analysis-ML-AI
pip install -r requirements.txt
```

### 🤝 Connect With Me
- **LinkedIn**: [Rotich Kibet](https://www.linkedin.com/in/rotichkibet/)
- **GitHub**: [Kibet-Rotich](https://github.com/Kibet-Rotich)

Thank you for visiting! Feel free to open issues, suggest improvements, or collaborate.
