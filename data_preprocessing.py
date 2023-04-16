import pandas as pd
import nltk
import pyrebase
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Initialize Firebase project
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

firebase = pyrebase.initialize_app(config)

# Download the NLTK stop words and wordnet dictionary
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset into a pandas dataframe and drop rows with missing values
df = pd.read_csv('C:\\Users\\Anand\\Desktop\\projectmodel.csv').dropna()

# Remove punctuation and convert to lowercase
df['clean_text'] = df['clean_text'].str.replace('[^\w\s]', '', regex=True).str.lower()


# Tokenize the text and remove stop words
stop_words = set(stopwords.words('english'))
df['tokens'] = df['clean_text'].apply(lambda x: [token for token in word_tokenize(x) if token not in stop_words])

# Lemmatize the tokens
lemmatizer = WordNetLemmatizer()
df['lemmas'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

# Get sentiment scores using VADER
sid = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x))

# Extract the compound sentiment score from the sentiment scores
df['compound_sentiment'] = df['sentiment_scores'].apply(lambda x: x['compound'])


# Create a reference to the Firebase Realtime Database
db = firebase.database()

# Upload preprocessed data to Firebase
for index, row in df.iterrows():
    data = {
        'clean_text': row['clean_text'],
        'tokens': row['tokens'],
        'lemmas': row['lemmas'],
        'compound_sentiment': row['compound_sentiment']
    }
    db.child('preprocessed_data').push(data)
