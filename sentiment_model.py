import openai
from transformers import pipeline

# set up OpenAI API key
openai.api_key = "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"

# set up sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# define function to get sentiment from input words
def get_sentiment(words):
    result = sentiment_pipeline(words)[0]
    return result["label"]

# example usage
words = "I love this product"
sentiment = get_sentiment(words)
print(sentiment)
