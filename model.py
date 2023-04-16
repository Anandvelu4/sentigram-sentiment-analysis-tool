import pandas as pd
import pyrebase
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


config = {
  "apiKey": "AIzaSyC3laasvWRbXuWUMJThljo8HP6fLtJxBG4",
  "authDomain": "sentimentanalysis-ec432.firebaseapp.com",
  "databaseURL": "https://sentimentanalysis-ec432-default-rtdb.firebaseio.com",
"  projectId": "sentimentanalysis-ec432",
  "storageBucket": "sentimentanalysis-ec432.appspot.com",
 " messagingSenderId": "627617192265",
 " appId": "1:627617192265:web:52c4ec14e8db68c1e96298",
 " measurementId": "G-133HWM4J7Z"
};

# Load the preprocessed dataset from Firebase
firebase = pyrebase.initialize_app(config)
db = firebase.database()
data = db.child("preprocessed_data").get().val()
df = pd.DataFrame.from_dict(data, orient='index')

# Drop rows with missing values from X_train
df.dropna(inplace=True)

# Convert lemmas back to strings
df['lemmas'] = df['lemmas'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

# Split the data into training and testing sets
X = df['lemmas'].values
y = np.where(df['compound_sentiment'] >= 0, 1, 0) # Convert the continuous sentiment score to binary labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract features using bag of words
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_bow, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(nb, X_train_bow, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# Evaluate the model on the test set
y_pred = nb.predict(X_test_bow)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))



# Save the trained model and feature extraction parameters using pickle
with open('pages/model.pkl', 'wb') as f:
    pickle.dump(nb, f)

with open('pages/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

