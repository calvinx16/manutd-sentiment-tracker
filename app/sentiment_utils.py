import pandas as pd
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze sentiment of a text using both TextBlob and VADER.

    Parameters:
    text (str): The input text (e.g., tweet or sentence).

    Returns:
    pd.Series: A pandas Series containing:
        - Polarity (float): Sentiment polarity from TextBlob (-1 to 1)
        - Subjectivity (float): Subjectivity score from TextBlob (0 to 1)
        - VADER_Compound (float): Compound sentiment score from VADER (-1 to 1)
        - Overall_Sentiment (str): Combined label: 'Positive', 'Neutral', or 'Negative'
    """
    blob = TextBlob(text)
    vader_scores = sia.polarity_scores(text)

    return pd.Series({
        'Polarity': blob.sentiment.polarity,
        'Subjectivity': blob.sentiment.subjectivity,
        'VADER_Compound': vader_scores['compound'],
        'Overall_Sentiment': classify_sentiment(blob.sentiment.polarity, vader_scores['compound'])
    })

def classify_sentiment(blob_polarity, vader_compound):
    """
    Classify sentiment based on both TextBlob polarity and VADER compound score.

    Parameters:
    blob_polarity (float): Polarity score from TextBlob.
    vader_compound (float): Compound score from VADER.

    Returns:
    str: 'Positive', 'Neutral', or 'Negative' sentiment label.
    """
    if blob_polarity > 0.1 and vader_compound > 0.1:
        return 'Positive'
    elif blob_polarity < -0.1 and vader_compound < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def preprocess_data(df):
    """
    Preprocess a DataFrame of tweets by computing sentiment metrics.

    Parameters:
    df (pd.DataFrame): A DataFrame containing a 'Tweet' column.

    Returns:
    pd.DataFrame: Original DataFrame with added sentiment analysis columns.
    """
    sentiment_scores = df['Tweet'].apply(analyze_sentiment)
    return pd.concat([df, sentiment_scores], axis=1)
