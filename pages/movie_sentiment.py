import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from textblob import TextBlob

# Set up TMDb API key
TMDB_API_KEY = "0afe20227bd7da8daddce2c9af601156"

# Define function to get movie reviews from TMDb API
def get_movie_reviews(movie_id):
    # Make request to TMDb API to get reviews for specified movie
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}&language=en-US&page=1")
    results = response.json()["results"]
    # Extract text content from reviews
    reviews = [review["content"] for review in results]
    return reviews

# Define function to perform sentiment analysis on movie reviews
def analyze_sentiment(reviews):
    # Initialize empty dataframe to store sentiment analysis results
    df = pd.DataFrame(columns=["Sentiment", "Count"])
    # Loop through reviews and perform sentiment analysis using TextBlob
    for review in reviews:
        blob = TextBlob(review)
        sentiment_score = round(blob.sentiment.polarity, 2)
        if sentiment_score > 0:
            sentiment = "Positive"
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        # Update dataframe with sentiment analysis result
        if sentiment in df["Sentiment"].values:
            index = df[df["Sentiment"] == sentiment].index[0]
            df.at[index, "Count"] += 1
        else:
            df = df.append({"Sentiment": sentiment, "Count": 1}, ignore_index=True)
    return df

# Create Streamlit app
def app():
    st.title("Movie Sentiment Analysis")
    st.write("Enter the names of two movies to perform sentiment analysis on their reviews and compare their sentiment scores.")

    # Get user input
    movie1_name = st.text_input("Enter first movie name")
    movie2_name = st.text_input("Enter second movie name")

    # Get TMDb movie IDs for user input using TMDb API search endpoint
    movie1_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie1_name}")
    movie2_response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie2_name}")
    movie1_results = movie1_response.json()["results"]
    movie2_results = movie2_response.json()["results"]
    if len(movie1_results) > 0:
        movie1_id = movie1_results[0]["id"]
    else:
        movie1_id = None
    if len(movie2_results) > 0:
        movie2_id = movie2_results[0]["id"]
    else:
        movie2_id = None

    # Get movie reviews and perform sentiment analysis for each movie
    if movie1_id and movie2_id:
        movie1_reviews = get_movie_reviews(movie1_id)
        movie2_reviews = get_movie_reviews(movie2_id)
        movie1_df = analyze_sentiment(movie1_reviews)
        movie2_df = analyze_sentiment(movie2_reviews)

        # Display sentiment analysis results in table and line chart
        st.write(f"## Sentiment analysis for {movie1_name}")
        st.write(movie1_df)
        fig1, ax1 = plt.subplots()
        ax1.plot(movie1_df["Sentiment"], movie1_df["Count"], marker='o', color='b')
        ax1.set_title(f"Sentiment analysis for {movie1_name}")
        ax1.set_xlabel("Sentiment")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        st.write(f"## Sentiment analysis for {movie2_name}")
        st.write(movie2_df)
        fig2, ax2 = plt.subplots()
        ax2.plot(movie2_df["Sentiment"], movie2_df["Count"], marker='o', color='b')
        ax2.set_title(f"Sentiment analysis for {movie2_name}")
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    else:
        st.write("Unable to retrieve movie information. Please check the spelling of the movie names and try again.")

if __name__ == "__main__":
    app()
