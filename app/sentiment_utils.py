import pandas as pd
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def preprocess_data(df):
    df['Sentiment'] = df['Tweet'].apply(analyze_sentiment)
    return df
