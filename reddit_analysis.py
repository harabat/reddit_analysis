# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reddit Analysis

# %%
from datetime import datetime
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pprint import pprint
import praw
from psaw import PushshiftAPI
import seaborn as sns

# %% [markdown]
# ## PSAW

# %%
# initialising api for pushshift and Reddit

reddit = praw.Reddit(
    client_id=os.environ['CLIENT_ID'],
    client_secret=os.environ['CLIENT_SECRET'],
    user_agent=os.environ['USER_AGENT'],
)

api = PushshiftAPI(reddit)

# %% [markdown]
# ## Submissions

# %% [markdown]
# ### Getting the data

# %%
# defining query parameters

start_epoch = int(datetime(2020, 1, 1).timestamp())
end_epoch = int(datetime(2021, 1, 1).timestamp())
subreddit = os.environ['SUBREDDIT']

# creating the query

submissions_gen = api.search_submissions(
    subreddit=subreddit,
    after=start_epoch,
    before=end_epoch
)

# %%
# getting the data and pickling it

submissions = list(submissions_gen)

with open('reddit_submissions', 'wb') as file:
    pickle.dump(submissions, file)

# %%
# unpickling the data

with open('reddit_submissions', 'rb') as file:
    submissions = pickle.load(file)

# %%
# populating dataframe, cleaning it up, and pickling it

data_sub = pd.DataFrame(
    item.__dict__ for item in list(submissions)
)

data_sub = data_sub.reindex(columns=sorted(data_sub.columns))

data_sub.to_pickle('reddit_data_sub')

# %% [markdown]
# ### Exploring the data

# %%
data_sub = pd.read_pickle('reddit_data_sub.pkl')

# %%
pprint(data_sub.columns)

# %%
# exploring submission data

data_sub['date'] = data_sub['created_utc'].apply(
    lambda x: datetime.utcfromtimestamp(x))

print(data_sub['date'])
print(sorted(data_sub['score'].unique(), reverse=True))
print(sorted(data_sub['num_comments'].unique(), reverse=True))

# %%
sns.lineplot(data=data_sub, x='date', y='score')

# %% [markdown]
# ## Comments

# %% [markdown]
# ### Getting the data

# %%
# no need for PRAW here, only PSAW
api = PushshiftAPI()

# %%
# defining query parameters
start_epoch = int(datetime(2020, 1, 1).timestamp())
end_epoch = int(datetime(2021, 1, 1).timestamp())
subreddit = os.environ['SUBREDDIT']

# creating the query
comments_gen = api.search_comments(
    subreddit=subreddit,
    after=start_epoch,
    before=end_epoch
)

# %%
# getting the data and pickling it
comments = []

for comment in comments_gen:
    comments.append(comment)
    if not len(comments) % 1000:
        print(len(comments))

# %%
# populating dataframe, cleaning it up, and pickling it
data_com = pd.DataFrame(
    item.d_ for item in comments
)

# add date and len columns
data_com_full['date'] = data_com_full['created_utc'].apply(
    lambda x: datetime.utcfromtimestamp(x))
data_com_full['len'] = data_com_full['body'].apply(len)

# sort columns alphabetically
data_com = data_com.reindex(columns=sorted(data_com.columns))

# pickle data
data_com.to_pickle('reddit_data_com.pkl')

# %% [markdown]
# ### Merging fragmented data and full data

# %% [markdown]
# #### Fragmented data

# %%
# retrieve data from pickle

data_com_part = pd.DataFrame()

for file in os.listdir('PSAW/'):
    filename = os.fsdecode(file)
    if filename.startswith('reddit_data_com_0'):
        print(filename)
        data_com_part = pd.read_pickle('PSAW/' + filename)
        data_com_part = data_com_part.append(data_com_part, ignore_index=True)

data_com_part.shape

# %%
# add date and len columns
data_com_part['date'] = data_com_part['created_utc'].apply(
    lambda x: datetime.utcfromtimestamp(x))
data_com_part['len'] = data_com_part['body'].apply(len)

# counting duplicates
duplicates = data_com_part.duplicated(
    subset=['created_utc', 'author', 'id']
)
data_com_part.drop_duplicates(
    subset=['created_utc', 'author', 'id'],
    inplace=True
)
print(sum(duplicates))

# cleaning the data
data_com_part = data_com_part.reindex(columns=sorted(data_com_part.columns))
data_com_part.sort_values('date', ascending=False, inplace=True)

# pickling data
data_com_part.to_pickle('reddit_data_com_part.pkl')

# %%
# exploring comments data

print(data_com_part.shape)
print(data_com_part['date'])

# %% [markdown]
# #### Exploring the full data

# %%
data_com_full = pd.read_pickle('reddit_data_com_full.pkl')
data_com_full.shape

# %%
# counting duplicates
duplicates = data_com_full.duplicated(
    subset=['created_utc', 'author', 'id']
)
data_com_full.drop_duplicates(
    subset=['created_utc', 'author', 'id'],
    inplace=True
)
print(sum(duplicates))

data_com_full.shape

# %%
# exploring comments data
print(data_com_full['date'])
print(sorted(data_com_full['score'].unique(), reverse=True))

# %% [markdown]
# #### Comparing fragmented data and full data

# %%
data_com_part = pd.read_pickle('reddit_data_com_part.pkl')
data_com_full = pd.read_pickle('reddit_data_com_full.pkl')

# %%
print(
    'data_com:\t',
    data_com_part.shape,
    '\ndata_com_full:\t',
    data_com_full.shape
)

# %%
data_com = data_com_part.append(data_com_full, ignore_index=True)

data_com['date'] = data_com['created_utc'].apply(
    lambda x: datetime.utcfromtimestamp(x))
data_com['len'] = data_com['body'].apply(len)

# %%
# counting duplicates
duplicates = data_com.duplicated(
    subset=['created_utc', 'author', 'id'],
    keep='first'
)

print(sum(duplicates))

data_com.drop_duplicates(
    subset=['created_utc', 'author', 'id'],
    keep='first',
    inplace=True
)

print(data_com.shape)

# %%
data_com.to_pickle('reddit_data_com.pkl')

# %% [markdown]
# ### Cleaning the data

# %%
# unpickle data
data_com = pd.read_pickle('reddit_data_com.pkl')

# %%
# clean data
data_com.fillna(value=np.nan, inplace=True)
data_com.mask(data_com.applymap(str).eq('[]'), other=np.nan, inplace=True)
data_com.replace(r'^\s*$', np.nan, regex=True, inplace=True)
data_com.dropna(axis=1, how='all', inplace=True)

data_com.columns

# %%
# only keep columns relevant columns
columns = [
    'author',  # author of comment
    'body',  # body of comment
    #     'created_utc', # date created (UTC) as epoch
    'date',  # date created (UTC)
    'distinguished',  # moderator comments
    'gildings',  # number of times comment was given gold
    'id',
    'len',  # len of body
    'link_id',  # submission id
    'parent_id',  # id of comment or submission being replied to
    'permalink',  # url of comment
    'score',  # score
]

data_com = data_com.reindex(columns=columns, copy=False)
data_com.columns

# %%
# standardize id
data_com['link_id'] = data_com['link_id'].apply(lambda x: str(x)[3:])
data_com['parent_id'] = data_com['parent_id'].apply(lambda x: str(x)[3:])

# %% [markdown]
# ### Most upvoted comments

# %%
data_com.sort_values('score', ascending=False)[:5]

# %%
for i in data_com.sort_values('score', ascending=False)[:5].index:
    for j in data_com.columns:
        print(data_com.iloc[i][j])

# %% [markdown]
# ### Comments that generated the most discussion

# %%
# link_id is the relevant submission
# parent_id is the parent item (comment or submission)
first_level = data_com['link_id'] == data_com['parent_id']

print(
    'First level comments:\t', sum(first_level),
    '\nReply comments:\t', sum(~first_level)
)

# %%
# initiliaze dict from first-level comment ids
most_active = dict.fromkeys(
    data_com[first_level]['id'],
    0
)

# loop over parent_id until every comment has been assigned to original first-level one
mask = [True] * len(data_com)
n = -1

while n != sum(most_active.values()):
    n = sum(most_active.values())

    ids = data_com[mask]['parent_id']
    mask = data_com['id'].isin(ids)

    for i in data_com[mask]['id']:
        try:
            most_active[i] += 1
        except KeyError:
            pass
        
# turn dict into dataframe
most_active = pd.DataFrame.from_dict(
    most_active,
    orient='index',
    columns=['Replies']
).sort_values('Replies', ascending=False)

most_active.head()

# %%
set(data_com[first_level]['id']) == set(most_active.index)

# %%
data_com[data_com['id'].isin(most_active[:5].index)]

# %% [markdown]
# ### How common is it to get replies to a comment?

# %%
replies_prop = pd.DataFrame(
    {
        'reply_count': range(most_active.values.min(), most_active.values.max() + 1),
        'proportion_less': [
            sum(most_active['Replies'] == i) / len(most_active) for i in range(
                most_active.values.min(), most_active.values.max() + 1
            )
        ]
    }
).set_index('reply_count')

replies_prop.plot.bar()
sns.barplot()

# %%
most_active

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(data=replies_prop, ax=ax)
# ax.

# %%
sns.histplot(most_active, log_scale=[False, True])

# %% [markdown]
# ### Members

# %%
moderators = data_com['distinguished'] == 'moderator'

not_moderators = data_com['distinguished'] != 'moderator'
sum(not_moderators)

# %%
members = sorted(data_com['author'].unique())
members[:10]

# %%
data_com.groupby('author').count()['distinguished'].sort_values()[-10:]

# %%
most_active_members = data_com.groupby('author').count()['score']
most_active_members.sort_values(ascending=False)

# %%
most_upvoted_members = data_com.groupby('author').sum()['score']
most_upvoted_members.sort_values(ascending=False)

# %%
data_members = pd.DataFrame({
    'members': members,
    'most_active': most_active_members,
    'most_upvoted': most_upvoted_members
})

data_members.describe()

# %% [markdown]
# #### Proportion of active members against subscribers

# %%
total_members = 360000

proportion = len(data_members) / total_members

print(
    'Proportion of active members vs total of subscribers:\t{:.1%}'.format(proportion)
)

# %% [markdown]
# ####  Most active members
#

# %%
data_members[data_members['most_active'] > 1000]['most_active'].hist()

# %% [markdown]
# #### Most upvoted members

# %%
sns.barplot(
    data=data_members,
    x='members',
    y='most_upvoted'
)

# %% [markdown]
# ## PRAW

# %% [markdown]
# ### Importing and pickling data

# %%
reddit = praw.Reddit(
    client_id=os.environ['CLIENT_ID'],
    client_secret=os.environ['CLIENT_SECRET'],
    user_agent=os.environ['USER_AGENT'],
)

# %%
# submissions = list(reddit.subreddit(os.environ['SUBREDDIT']).top(limit=1000))

# comments = [i.comments for i in submissions]

# with open('PRAW/reddit_submissions', 'wb') as file:
#     pickle.dump(submissions, file)

# with open('PRAW/reddit_comments', 'wb') as file:
#     pickle.dump(comments, file)

# %%
# with open('PRAW/reddit_submissions', 'rb') as file:
#     submissions = pickle.load(file)

# with open('PRAW/reddit_comments', 'rb') as file:
#     comments = pickle.load(file)

# %%
# data_sub = pd.DataFrame(
#     item.__dict__ for item in submissions
# )

# data_sub = data_sub.reindex(columns=sorted(data_sub.columns))

# data_sub.to_pickle('PRAW/reddit_data_sub')

# data_sub.head()

# %%
# data_com = pd.DataFrame(
#     comment.__dict__ for item in comments for comment in item.list()
# )

# data_com = data_com.reindex(columns=sorted(data_com.columns))

# data_com.to_pickle('PRAW/reddit_data_com')

# data_com.head()

# %% [markdown]
# ### Checking validity of data

# %%
data_sub = pd.read_pickle('PRAW/reddit_data_sub')
data_com = pd.read_pickle('PRAW/reddit_data_com')

# %% [markdown]
# ### Timing

# %%
days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

data_sub['Day'] = data_sub['created_utc'].apply(
    lambda x:
    days[
        datetime.utcfromtimestamp(x).weekday()
    ]
)

data_sub['Hour'] = data_sub['created_utc'].apply(
    lambda x:
        datetime.utcfromtimestamp(x).hour
)

data_sub

# %%
data_com = data_com[data_com['created_utc'] > 0]

days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

data_com['Day'] = data_com['created_utc'].apply(
    lambda x:
    days[
        datetime.utcfromtimestamp(x).weekday()
    ]
)

data_com['Hour'] = data_com['created_utc'].apply(
    lambda x:
        datetime.utcfromtimestamp(x).hour
)

data_com


# %%
def median(x):
    if len(x) > 10:
        return np.median(x)


def mean(x):
    if len(x) > 10:
        return np.mean(x)


# %% [markdown]
# #### Highest posting activity

# %%
posting_activity = data_sub.pivot_table(
    index='Day',
    columns='Hour',
    values='score',
    aggfunc=len
).reindex(index=days)

posting_activity

# %%
fig, ax = plt.subplots(figsize=(15, 5))

sns.heatmap(
    posting_activity,
    annot=True,
    fmt='.0f',
    cmap='YlOrBr',
    cbar=False,
    square=False,
    ax=ax
)

ax.set_title(
    'Number of submissions\ndepending on day of week and hour of posting',
    fontsize='x-large',
    weight='semibold'
)

# %% [markdown]
# #### Most commenting activity

# %%
commenting_activity = data_com.pivot_table(
    index='Day',
    columns='Hour',
    values='created_utc',
    aggfunc=len
).reindex(index=days)

commenting_activity

# %%
fig, ax = plt.subplots(figsize=(15, 5))

sns.heatmap(
    commenting_activity,
    annot=True,
    fmt='.0f',
    cmap='YlOrBr',
    cbar=False,
    square=False,
    ax=ax
)

ax.set_title(
    'Number of comments\ndepending on day of week and hour',
    fontsize='x-large',
    weight='semibold'
)

# %% [markdown]
# #### Commenting to posting ratio

# %%
ratio = commenting_activity / posting_activity

# %%
fig, ax = plt.subplots(figsize=(15, 5))

sns.heatmap(
    ratio,
    annot=True,
    fmt='.0f',
    cmap='YlOrBr',
    cbar=False,
    square=False,
    ax=ax
)

ax.set_title(
    'Commenting to posting ratio\ndepending on day of week and hour',
    fontsize='x-large',
    weight='semibold'
)

# %% [markdown]
# #### Maximising comments

# %% [markdown]
# ##### By day of week

# %%
data_sub.groupby('Day')['num_comments'].mean().reindex(index=days).plot()
data_sub.groupby('Day')['num_comments'].median().reindex(
    index=days).plot(color='red')

# %% [markdown]
# ##### By hour

# %%
data_sub.groupby('Hour')['num_comments'].mean().plot()
data_sub.groupby('Hour')['num_comments'].median().plot(color='red')

# %% [markdown]
# ##### By day of week and hour

# %%
timing_median_comments = data_sub.pivot_table(
    index='Day',
    columns='Hour',
    values='num_comments',
    aggfunc=mean
).reindex(index=days)

timing_median_comments

# %%
fig, ax = plt.subplots(figsize=(15, 5))

sns.heatmap(
    timing_median_comments,
    annot=True,
    fmt='.0f',
    cmap='YlOrBr',
    cbar=False,
    square=False,
    ax=ax
)

ax.set_title(
    'Median comments per submission\ndepending on day of week and hour of posting',
    fontsize='x-large',
    weight='semibold'
)

# %% [markdown]
# #### Maximising score

# %% [markdown]
# ##### By day of week

# %%
data_sub.groupby('Day')['score'].mean().reindex(index=days).plot()
data_sub.groupby('Day')['score'].median().reindex(index=days).plot(color='red')

# %% [markdown]
# ##### By hour

# %%
data_sub.groupby('Hour')['score'].mean().plot()
data_sub.groupby('Hour')['score'].median().plot(color='red')

# %% [markdown]
# #### By day of week and hour

# %%
timing_median_score = data_sub.pivot_table(
    index='Day',
    columns='Hour',
    values='score',
    aggfunc=mean
).reindex(index=days)

timing_median_score

# %%
fig, ax = plt.subplots(figsize=(15, 5))

sns.heatmap(
    timing_median_score,
    annot=True,
    fmt='.0f',
    cmap='YlOrBr',
    cbar=False,
    square=False,
    ax=ax
)

ax.set_title(
    'Median score per submission\ndepending on day of week and hour of posting',
    fontsize='x-large',
    weight='semibold'
)

# %%
sns.histplot(data_sub[(data_sub['Day'] == 'Thur') & (
    data_sub['num_comments'] >= 10)]['Hour'], bins=24)

# %% [markdown]
# ### Correlations

# %%
data_sub.drop(
    columns=[
        'allow_live_comments',
        'all_awardings',  # related to awards received
        'approved_by',  # indicates comments approved or banned by moderators
        'archived',
        'approved_at_utc',  # indicates comments approved or banned by moderators
        'author_cakeday',  # account creation anniversary
        'author_flair_background_color',  # related to flair
        'author_flair_css_class',  # related to flair
        'author_flair_richtext',  # related to flair
        'author_flair_template_id',  # related to flair
        'author_flair_text',  # related to flair
        'author_flair_text_color',  # related to flair
        'author_flair_type',  # related to flair
        'author_fullname',  # unique id for author
        'author_patreon_flair',  # related to flair
        #         'author_premium', # whether account is premium
        'awarders',  # related to awards received, mostly empty
        'banned_at_utc',  # indicates comments approved or banned by moderators
        'banned_by',  # indicates comments approved or banned by moderators
        'can_gild',
        'can_mod_post',
        'category',
        'clicked',
        'comment_limit',
        'comment_sort',
        'content_categories',
        'contest_mode',
        'created',  # equivalent to created_utc
        #         'created_utc', # date created (UTC)
        'crosspost_parent',
        'crosspost_parent_list',
        'discussion_type',
        #         'distinguished', # identifies moderators
        'edited',  # indicates edited comments
        'flair',
        'gilded',
        'gildings',  # number of times comment was given gold
        'hidden',
        'hide_score',
        'id',  # unique id for comment
        'is_crosspostable',
        'is_meta',  # empty
        'is_original_content',  # empty
        'is_robot_indexable',
        'is_submitter',  # whether commented is also submitter
        'is_video',  # empty
        'likes',  # empty
        'link_flair_background_color',  # no added information
        'link_flair_css_class',  # no added information
        'link_flair_richtext',  # no added information
        'link_flair_template_id',  # no added information
        'link_flair_text_color',  # no added information
        'link_flair_type',  # no added information
        #         'link_id', # unique id for submission
        'locked',  # identifies deleted messages
        'media_embed',  # same as media
        'media_metadata',  # same as media
        'media_only',  # empty
        'mod',  # empty
        'mod_note',  # empty
        'mod_reason_by',  # empty
        'mod_reason_title',  # empty
        'mod_reports',  # empty
        'no_follow',
        #         'num_crossposts', # no information
        #         'num_duplicates', # no information
        'num_reports',  # empty
        'over_18',  # empty
        #         'parent_id', # unique id of comment or submission being replied to
        'parent_whitelist_status',  # same value
        'permalink',  # url of comment
        'pinned',  # no added information
        'poll_data',  # few values
        'post_hint',  # few values
        'pwls',  # same value
        'quarantine',  # no added information
        'removal_reason',  # empty
        'removed_by',  # empty
        'removed_by_category',  # empty
        'report_reasons',  # empty
        'retrieved_on',  # date that the comment was scraped by pushshift
        'saved',  # empty
        #         'score', # score
        'secure_media',  # same as media
        'secure_media_embed',  # same as media
        'send_replies',  # mainly indicates archived threads, removed comments, AutoModerator comments
        'selftext_html',  # same as selftext
        'subreddit',  # related to subreddit
        'subreddit_id',  # unique id for subreddit
        'subreddit_name_prefixed',  # related to subreddit
        'subreddit_subscribers',  # related to subreddit
        'subreddit_type',  # related to subreddit
        'send_replies',  # few values
        'spoiler',  # not relevant
        'steward_reports',  # no information
        'stickied',  # moderator comments
        'suggested_sort',  # few values
        'thumbnail',  # depends on domain
        'thumbnail_height',  # no added information
        'thumbnail_width',  # no added information
        'total_awards_received',  # total awards received
        'top_awarded_type',  # empty
        'treatment_tags',  # no information, empty
        'url_overridden_by_dest',  # same as 'domain' not self,  'self' == False
        'user_reports',  # empty
        'view_count',  # empty
        'visited',  # empty
        'whitelist_status',  # same value
        'wls',  # same value
        '_comments',
        '_comments_by_id',
        '_fetched',
        '_reddit'
    ],
    errors='ignore',
    inplace=True
)

data_sub.head()

# %%
# categories that should have correlations

# self-post
data_sub[data_sub['selftext'].apply(len) > 0]['domain'].value_counts()
# title-only self-post
data_sub[data_sub['is_self'] == True]['domain'].value_counts()

# reddit media
data_sub[data_sub['is_reddit_media_domain'] == True]['domain'].value_counts()

# non-reddit media
data_sub[data_sub['media'].apply(bool) == True]['domain'].value_counts()

# %%
# make dummy variables here?

data_sub.replace(
    to_replace={
        False: 0,
        True: 1,
        np.NaN: 0,
        None: 0,
        '': 0
    },
    inplace=True
)

data_sub.replace(
    to_replace='.+',
    value=1,
    regex=True,
    inplace=True
)

data_sub.describe().transpose()

# %%
# pairwise correlation table

corr_data_sub = data_sub.corr().dropna(how='all').dropna(axis=1, how='all')

corr_data_sub.head()

# %% [raw]
# # mask for strong correlations
#
# mask = corr_data_sub.applymap(lambda x: abs(x) > 0.45 and x < 1)
#
# mask.head()

# %%
# plot of strong correlations

fig, ax = plt.subplots(
    figsize=(10, 5)
)

sns.heatmap(
    corr_data_sub[mask].dropna(how='all').dropna(axis=1, how='all'),
    annot=True,
    fmt='.1f',
    linewidths=.5,
    cmap='YlOrBr',
    cbar=False,
    #     square=True,
    ax=ax
)

ax.set_xticklabels(
    labels=ax.get_xticklabels(),
    ha='right',
    rotation=30
)

ax.set_title(
    'Most correlated values',
    fontsize='x-large',
    weight='semibold'
)

# %%
# mask for weak correlations

mask_inverse = corr_data_sub.applymap(lambda x: abs(x) < 0.45)

mask_inverse.head()

# %%
# plot of weak correlations

fig, ax = plt.subplots(
    figsize=(15, 10)
)

sns.heatmap(
    corr_data_sub[mask_inverse].dropna(how='all').dropna(axis=1, how='all'),
    annot=True,
    fmt='.1f',
    linewidths=.5,
    cmap='YlOrBr',
    cbar=False,
    ax=ax
)

ax.set_xticklabels(
    labels=ax.get_xticklabels(),
    ha='right',
    rotation=30
)

ax.set_title(
    'Least correlated values',
    fontsize='x-large',
    weight='semibold'
)

# %% [markdown]
# ### Titles

# %%
# submissions = list(reddit.subreddit(os.environ['SUBREDDIT']).top(limit=1000))

# comments = [i.comments for i in submissions]

# with open('PRAW/reddit_submissions_top', 'wb') as file:
#     pickle.dump(submissions, file)

# with open('PRAW/reddit_comments_top', 'wb') as file:
#     pickle.dump(comments, file)

with open('PRAW/reddit_submissions_top', 'rb') as file:
    submissions_top = pickle.load(file)

with open('PRAW/reddit_comments_top', 'rb') as file:
    comments_top = pickle.load(file)

# %%
data_sub_top = pd.DataFrame(
    i.__dict__ for i in submissions_top
)

data_sub_top = data_sub_top.reindex(columns=sorted(data_sub_top.columns))

data_sub_top.to_pickle('PRAW/reddit_data_sub_top')

data_sub_top.head(2)

# %%
data_com_top = pd.DataFrame(
    comment.__dict__ for item in comments_top for comment in item.list()
)

data_com_top = data_com_top.reindex(columns=sorted(data_com_top.columns))

data_com_top.to_pickle('PRAW/reddit_data_com_top')

data_com_top.head(2)

# %%
languages = pd.read_csv(
    './languages.csv')

languages.head()

# %% [markdown]
# #### Comments

# %%
comments_full = ' '.join(list(str(i).lower() for i in data_com_top['body']))

comments_tokenize = findall('[a-z]+', comments_full)

# %%
freq = {}

for language in languages['Language']:
    freq_language = comments_tokenize.count(language.lower())
    freq[language] = freq_language
    if language.find('/') > -1:
        freq[language] += comments_tokenize.count(
            language.split('/')[1].lower())

# %%
languages_freq = pd.DataFrame(
    {
        'Language': freq.keys(),
        'Count': freq.values()
    }
).sort_values('Count', ascending=False, ignore_index=True)

languages_freq['Percentage'] = languages_freq['Count'] / \
    languages_freq['Count'].sum() * 100

languages_freq

# %%
sns.barplot(x='Count', y='Language',
            data=languages_freq[languages_freq['Percentage'] > 1])

# %% [markdown]
# #### Submissions

# %%
submissions_full = ' '.join(list(str(i).lower() for i in (
    data_sub_top['title'] + ' ' + data_sub_top['selftext'])))

submissions_tokenize = findall('[a-z]+', submissions_full)

# %%
freq = {}

for language in languages['Language']:
    freq_language = submissions_tokenize.count(language.lower())
    freq[language] = freq_language
    if language.find('/') > -1:
        freq[language] += submissions_tokenize.count(
            language.split('/')[1].lower())

# %%
languages_freq = pd.DataFrame(
    {
        'Language': freq.keys(),
        'Count': freq.values()
    }
).sort_values('Count', ascending=False, ignore_index=True)

languages_freq['Percentage'] = languages_freq['Count'] / \
    languages_freq['Count'].sum() * 100

languages_freq

# %%
sns.barplot(x='Count', y='Language',
            data=languages_freq[languages_freq['Percentage'] > 1])
