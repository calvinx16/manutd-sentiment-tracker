import snscrape.modules.twitter as sntwitter
import pandas as pd


def scrape_tweets(query, limit=100):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        tweets.append([tweet.date, tweet.content])

    df = pd.DataFrame(tweets, columns=['Date', 'Tweet'])
    return df