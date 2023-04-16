import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved model and feature extraction parameters using pickle
with open('pages/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pages/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Define a function to classify the sentiment of user input
def classify_sentiment(text):
    # Extract features using bag of words
    text_bow = vectorizer.transform([text])

    # Predict the sentiment using the trained model
    sentiment = model.predict(text_bow)[0]

    # Return the sentiment label
    return "Positive" if sentiment == 1 else "Negative"


# Create the Streamlit app
st.title("Sentiment Analysis App")

# Add a text input for user to enter text
text = st.text_input("Enter some text:")

# Check if user has entered any text
if text:
    # Classify the sentiment of the text
    sentiment = classify_sentiment(text)

    # Display the sentiment label
    st.write(f"The sentiment of the text is {sentiment}.")
