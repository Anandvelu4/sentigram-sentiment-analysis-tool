import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# News API endpoint
news_api_endpoint = 'https://newsapi.org/v2/everything'

# Your News API key
news_api_key = '29b03da959fd4801b1479a98eaf9ba4e'

# Streamlit app title and configuration
st.set_page_config(page_title='Sentiment Analysis', page_icon=':bar_chart:', layout='wide')

# Streamlit app header
st.title('Sentiment Analysis of News Articles')

# User input for topic to search for
search_topic = st.text_input('Enter a topic to search for:', 'Tesla')

# Perform API request to get news articles related to the topic
params = {
    'q': search_topic,
    'sortBy': 'relevancy',
    'apiKey': news_api_key
}
response = requests.get(news_api_endpoint, params=params).json()

# Check if there are any articles returned
if response['totalResults'] == 0:
    st.warning(f"No articles found for '{search_topic}'")
else:
    # Convert articles into a Pandas DataFrame
    articles = response['articles']
    articles_df = pd.DataFrame(articles, columns=['title', 'description', 'url', 'publishedAt'])

    # Preprocess the text data by lowercasing, removing punctuations and stop words
    stop_words = stopwords.words('english')
    articles_df['cleaned_text'] = articles_df['title'].str.lower().str.replace('[^\w\s]', '').apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Calculate sentiment score for each article using TextBlob
    articles_df['sentiment_score'] = articles_df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Categorize each article as positive, negative or neutral based on sentiment score
    articles_df['sentiment_category'] = articles_df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

    # Calculate the weightage of each sentiment category
    sentiment_weightage = articles_df['sentiment_category'].value_counts(normalize=True)

    # Plot a bar chart of sentiment weightage
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=sentiment_weightage.index, y=sentiment_weightage.values, ax=ax)
    ax.set_title(f"Sentiment Analysis of {search_topic}")
    ax.set_xlabel('Sentiment Category')
    ax.set_ylabel('Weightage')
    st.pyplot(fig)

    # Display top 3 positive and negative articles
    st.subheader("Top 3 Positive Articles")
    positive_articles = articles_df[articles_df['sentiment_category'] == 'positive'].head(3)
    for index, row in positive_articles.iterrows():
        st.write(f"- {row['title']} ({row['publishedAt']})")

    st.subheader("Top 3 Negative Articles")
    negative_articles = articles_df[articles_df['sentiment_category'] == 'negative'].head(3)
    for index, row in negative_articles.iterrows():
        st.write(f"- {row['title']} ({row['publishedAt']})")

        # Display emoji rank based on sentiment weightage
    if sentiment_weightage.idxmax() == 'positive':
        st.write(":smiley: Positive sentiment won!")
    elif sentiment_weightage.idxmax() == 'negative':
        st.write(":disappointed: Negative sentiment won!")
    else:
        st.write(":neutral_face: Neutral sentiment won!")
