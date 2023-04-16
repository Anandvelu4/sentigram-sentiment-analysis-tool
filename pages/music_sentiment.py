import streamlit as st
from textblob import TextBlob
import requests
import json

# Spotify API credentials
CLIENT_ID = "47d1e6083d684bd2ae7918e81b4c1015"
CLIENT_SECRET = "8c35c7ee0a0842538791118b1ac2978e"


# Define function to get Spotify access token
def get_access_token():
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    token = json.loads(response.text)["access_token"]
    return token


# Define function to get music recommendation based on sentiment
def get_music_recommendation(sentiment):
    # Get Spotify access token
    access_token = get_access_token()

    # API endpoint for music recommendation service
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {access_token}"}

    # Map sentiment to music genre
    if sentiment == "positive":
        seed_genres = ["pop", "dance", "electronic"]
    elif sentiment == "negative":
        seed_genres = ["rock", "metal"]
    else:
        seed_genres = ["classical"]

    # Send request to Spotify API to get recommended tracks
    params = {"limit": 1, "seed_genres": ",".join(seed_genres)}
    response = requests.get(url, headers=headers, params=params)
    recommendation = json.loads(response.text)["tracks"][0]["name"]
    return recommendation


# Create Streamlit app
def app():
    st.title("Sentiment Analysis and Music Recommendation")
    st.write("Answer a few questions about your current sentiment and we'll recommend some music for you.")

    # Get user input
    mood = st.selectbox("What's your current mood?", ["Happy", "Sad", "Angry", "Neutral"])
    activity = st.selectbox("What are you doing?", ["Working", "Exercising", "Studying", "Relaxing"])

    # Map user input to sentiment label
    if mood == "Happy":
        sentiment = "positive"
    elif mood == "Sad":
        sentiment = "negative"
    elif mood == "Angry":
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Display sentiment analysis results
    st.write(f"Your sentiment is {sentiment}.")

    # Get music recommendation based on sentiment
    recommendation = get_music_recommendation(sentiment)

    # Display music recommendation
    st.write(f"Here's a song recommendation for {activity}: {recommendation}")

    # Add sentiment analysis for paragraph input
    st.write("You can also enter a paragraph to analyze its sentiment.")
    paragraph = st.text_area("Enter paragraph here")
    if st.button("Analyze"):
        # Use TextBlob to analyze sentiment
        blob = TextBlob(paragraph)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiment_label = "positive"
        elif sentiment < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Display sentiment analysis results
        st.write(f"The sentiment of the paragraph is {sentiment_label}.")

        # Get music recommendation based on sentiment
        recommendation = get_music_recommendation(sentiment_label)

        # Display music recommendation
        st.write(f"Here's a song recommendation for {activity}: {recommendation}")


if __name__ == "__main__":
    app()
