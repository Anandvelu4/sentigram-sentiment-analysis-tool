import streamlit as st
import wikipedia
from textblob import TextBlob
import altair as alt
import pandas as pd


# create Streamlit app
def app():
    st.title("Famous People Sentiment Analysis")
    st.write("Enter a famous person's name to get sentiment analysis on their Wikipedia page.")

    # take user input for person's name
    name = st.text_input("Person's Name", value="Barack Obama")

    try:
        # retrieve person's Wikipedia page
        page = wikipedia.page(name)

        # perform sentiment analysis on page content
        blob = TextBlob(page.content)
        sentiment = blob.sentiment

        # create visualization of sentiment analysis
        chart_data = {'Sentiment': ['Polarity', 'Subjectivity'], 'Score': [sentiment.polarity, sentiment.subjectivity]}
        chart_df = pd.DataFrame(chart_data)
        chart = alt.Chart(chart_df).mark_bar().encode(
            x='Sentiment',
            y='Score',
            color=alt.Color('Sentiment', scale=alt.Scale(scheme='dark2'))
        )
        st.write(f"Sentiment analysis for {name}:")
        st.altair_chart(chart, use_container_width=True)

    except wikipedia.exceptions.PageError:
        st.write(f"Sorry, no Wikipedia page found for {name}.")
    except wikipedia.exceptions.DisambiguationError:
        st.write(f"More than one Wikipedia page found for {name}. Please be more specific.")


# run the app
if __name__ == "__main__":
    app()
