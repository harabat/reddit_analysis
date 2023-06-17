# Reddit analysis

> **Warning**
>
> Following Reddit API changes, this code will soon no longer work, as access to Reddit data is being severely restricted.
>
> Find out about [the API changes](https://en.wikipedia.org/wiki/Reddit#2023_API_changes) and [the protests](https://www.theverge.com/2023/6/5/23749188/reddit-subreddit-private-protest-api-changes-apollo-charges) that it's generating.
>
> Alternatives to Reddit are [Lemmy](https://lemmy.world/) and [Kbin](https://kbin.social/).

This is an in-depth analysis of a subreddit and contains the following analyses:
- submissions
  - submissions by score
  - submissions by number of comments
- comments
  - comments by score
  - comments by number of subcomments
  - likelihood of getting replies to a comment
- subreddit members
  - most upvoted subreddit members
  - most active subreddit members by submissions
  - most active subreddit members by comments
  - proportion of active members vs subscribers
- trends over time
  - submissions over time
  - best time to post submissions by day of week and by hour
  - comments over time
  - best time to post comments by day of week and by hour
  - ratio of comments vs submissions
- correlations
    - whether posts with images are more likely to get upvotes, whether links to specific websites are more likely to get upvotes, etc.
