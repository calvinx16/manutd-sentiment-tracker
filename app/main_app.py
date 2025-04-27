import streamlit as st
import pandas as pd
import altair as alt
import os
from sentiment_utils import preprocess_data
from scrape import scrape_tweets

st.title("📊 Manchester United Fan Sentiment Tracker")

data_source = st.radio(
    "Choose data source:",
    ('Upload CSV', 'Scrape Live Tweets')
)

if data_source == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Tweet' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Custom file uploaded successfully!")
    else:
        st.info("No file uploaded. Using default sample tweets.")
        data_path = os.path.join('data', 'sample_tweets.csv')
        df = pd.read_csv(data_path)

elif data_source == 'Scrape Live Tweets':
    search_query = st.text_input("Enter search query (example: 'Manchester United since:2024-04-21 until:2024-04-22')", "Manchester United since:2024-04-21 until:2024-04-22")
    limit = st.slider("Number of tweets to scrape", 10, 500, 100)

    if st.button("Scrape Tweets"):
        df = scrape_tweets(search_query, limit)
        st.success(f"Scraped {len(df)} tweets!")

if 'df' in locals():
    st.write("Original Data", df.head())

    processed_df = preprocess_data(df)
    st.write("Processed Data with Sentiment", processed_df.head())

    sentiment_counts = processed_df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    color_scale = alt.Scale(
        domain=["Positive", "Neutral", "Negative"],
        range=["#DA291C", "#FBE122", "#000000"]
    )

    chart = alt.Chart(sentiment_counts).mark_bar().encode(
        x='Sentiment',
        y='Count',
        color=alt.Color('Sentiment', scale=color_scale)
    ).properties(
        title="Fan Sentiment Distribution (Manchester United Colors)"
    )

    st.altair_chart(chart, use_container_width=True)
