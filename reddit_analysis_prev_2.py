#!/usr/bin/python3

from datetime import datetime
import dotenv
import os
import pandas as pd
import praw
import seaborn as sns

dotenv.load_dotenv(dotenv.find_dotenv())

reddit = praw.Reddit(
    client_id=os.environ["CLIENT_ID"],
    client_secret=os.environ["CLIENT_SECRET"],
    user_agent=os.environ["USER_AGENT"],
)

submissions = list(reddit.subreddit(os.environ["SUBREDDIT"]).top(limit=1000))

data = pd.DataFrame(
    {
        "Date": [datetime.utcfromtimestamp(i.created_utc) for i in submissions],
        "Title": [i.title for i in submissions],
        "Hour": [datetime.utcfromtimestamp(i.created_utc).hour for i in submissions],
        "Day": [
            datetime.utcfromtimestamp(i.created_utc).weekday() for i in submissions
        ],
        "Score": [i.score for i in submissions],
        "original_content": [i.is_original_content for i in submissions],
        "reddit_media_domain": [i.is_reddit_media_domain for i in submissions],
        "self": [i.is_self for i in submissions],
        "video": [i.is_video for i in submissions],
        "url": [i.url for i in submissions],
    }
)

sns.histplot(data["Hour"])
