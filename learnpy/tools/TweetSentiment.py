__author__ = 'Jiarui Xu'

import textblob
import requests
import tweepy
from pylab import *

class TweetSentiment:
    """
    A tool designed for tweet sentiment analysis
    """
    def __init__(self, topic):
        self.make_pie(self.get_stats(topic), topic)

    def get_stats(self, topic):
        """
        Get sentiments analysis of the tweets
        :param topic:
        :return:
        """
        consumer_key = 'FQvMGcXfTBW085e9p8HrBGG03'
        consumer_secret = 'EyELEm6tZsgLK3HDuLYOVAtiKprCPNzx2YxOfllBvV3xZG5KSs'
        access_token = '4165420933-biz0Ne5cAEiMm7FivgPfPRu4oCIBYrf0UXOpAKI'
        access_token_secret = '6SYcS2b5zzek72huIqEK6njL1nz5cKtlFI0pBzuU4tiWx'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)

        results = api.search(q=topic, rpp = 100)

        # init stats
        pos = 0
        neg = 0
        for t in results:
            score = textblob.TextBlob(t.text).sentiment.polarity
            if score > 0:
                pos += 1
            elif score < 0:
                neg += 1
        return float(pos)/(pos+neg)

    def make_pie(self, percent, topic):
        """
        Make a pie chart
        :param percent:
        :return:
        """
        print ("The Positive Tweet Proportion is ", percent)
        figure(1, figsize=(6,6))
        ax = axes([0.1, 0.1, 0.8, 0.8])

        labels = 'Positive', 'Negative'
        fracs = [100*percent, 100.-100*percent]
        explode=(0.05, 0)

        pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

        title_con = topic + " tweets sentiment analysis"
        title(title_con, bbox={'facecolor':'0.8', 'pad':5})
        savefig('pie2.png')