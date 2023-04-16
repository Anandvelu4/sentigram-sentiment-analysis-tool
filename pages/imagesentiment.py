import openai
import streamlit as st
from transformers import pipeline

# set up OpenAI API key
openai.api_key = "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"

# set up sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# define function to generate image based on input words and sentiment
def generate_image(words, sentiment):
    prompt = f"Generate a cartoon character based on the words: {words} and the sentiment: {sentiment}"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    image_url = response.choices[0].text.strip()
    st.image(image_url)

# create streamlit app
def app():
    st.title("Cartoon Character Generator")
    st.write("Enter three words to generate a cartoon character based on them.")
    word1 = st.text_input("Word 1")
    word2 = st.text_input("Word 2")
    word3 = st.text_input("Word 3")
    if st.button("Generate Character"):
        words = f"{word1}, {word2}, {word3}"
        sentiment = sentiment_pipeline(words)[0]["label"]
        generate_image(words, sentiment)

if __name__ == "__main__":
    app()
