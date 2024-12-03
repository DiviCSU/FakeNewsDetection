#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


# In[ ]:


class SVMClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = SVC(kernel='linear', probability=True)
    
    def preprocess(self, data):
        X = self.vectorizer.fit_transform(data['Content'])
        y = data['Label']
        return X, y
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy:: {accuracy}")
        print("Classification Report:\n", report)
    
    def predict(self, texts):
        X_texts = self.vectorizer.transform(texts)
        return self.model.predict(X_texts)
        
    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
    def load_model(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)


# In[ ]:


# Load datasets
false_data = pd.read_csv("Cleaned_False_Data.csv")
true_data = pd.read_csv("Cleaned_True_Data.csv")

# Combine and label data
false_data['Label'] = 0  # Label for false articles
true_data['Label'] = 1   # Label for true articles
combined_data = pd.concat([false_data, true_data], ignore_index=True)
combined_data = combined_data[['Content', 'Label']].dropna()


# In[ ]:


# Instantiate the classifier
svm_classifier = SVMClassifier()

# Preprocess data
X, y = svm_classifier.preprocess(combined_data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
svm_classifier.train(X_train, y_train)

# Evaluate the classifier
svm_classifier.evaluate(X_test, y_test)


# In[ ]:


# Test predictions
new_articles = [
    "The government announced a new policy today.",
    "Aliens have landed, claims unverified report.",
]
predictions = svm_classifier.predict(new_articles)
for article, label in zip(new_articles, predictions):
    print(f"Article: {article}\nPrediction: {'True' if label == 1 else 'False'}\n")

# Save the model and vectorizer
svm_classifier.save_model("svm_model.pkl", "vectorizer.pkl")


# In[ ]:




