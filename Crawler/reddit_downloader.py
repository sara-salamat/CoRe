import datetime as dt
from itertools import count

import numpy as np
import pandas as pd
import praw
from praw.models import MoreComments
from psaw import PushshiftAPI

# save_path = pathlib.Path.home() / 'Desktop' / 'data2'
reddit = praw.Reddit(client_id='',
                    client_secret='',
                    user_agent='')


api = PushshiftAPI()
reddit.read_only = True
subreddit = 'changemyview'
ts_after = int(dt.datetime(2021, 1, 1).timestamp()) # start date
ts_before = int(dt.datetime(2022, 5, 1).timestamp()) # finish date

Posts_dict = {
    'title': [],
    'text' : [],
    'time' : [],
    'score': [], #upvotes
    'user_id': [],
    'post_id': [],
    'post_flair': [],
}
# using PSAW to search for posts
posts_gen = api.search_submissions(
            after=ts_after,
            before=ts_before,
            filter=['id'],
            subreddit=subreddit,
            limit=1000000000,
            # limit=10,
        )
print('Search completed!')
counter = 0
for post_psaw in posts_gen:
    submission_id = post_psaw.d_['id']
    post = reddit.submission(id=submission_id)
    counter = counter + 1
    # print('downloading post '+str(submission_id))
    if counter % 1000 == 0:
        print(str(counter) + ' posts downloaded!')

    if post.title == '[deleted by user]':
        continue
    if post.removed_by_category == 'moderator':
        continue
    if post.selftext == '[removed]' or post.selftext == '[deleted]':
        continue
    Posts_dict['title'].append(post.title)
    Posts_dict['text'].append(post.selftext)
    Posts_dict['time'].append(post.created)
    Posts_dict['score'].append(post.score)
    Posts_dict['user_id'].append(post.author)
    Posts_dict['post_id'].append(post.id)
    Posts_dict['post_flair'].append(post.link_flair_text)
print('Downloading posts completed!')
pd.DataFrame(Posts_dict).to_csv('posts.csv', index=False)

Comments_dict = {
    'post_id': [],
    'user_id': [],
    'tag': [],
    'time': [],
    'text': [],
    'comment_id': [],
    'delta': [],
}
counter = 0
for post_id in Posts_dict['post_id']:
    submission = reddit.submission(id=post_id)
    counter = counter + 1
    if counter % 1000 == 0:
        print('comments for ' + str(counter) + ' posts downloaded!')
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue
        Comments_dict['post_id'].append(top_level_comment.parent_id)
        Comments_dict['user_id'].append(top_level_comment.author)
        Comments_dict['time'].append(top_level_comment.created)
        Comments_dict['delta'].append(top_level_comment.author_flair_text)
        if top_level_comment.is_submitter:
            Comments_dict['tag'].append('OP')
        else:
            Comments_dict['tag'].append(None)
        Comments_dict['text'].append(top_level_comment.body)
        Comments_dict['comment_id'].append(top_level_comment.id)
pd.DataFrame(Comments_dict).to_csv('comments.csv', index=False)

list_of_users = Comments_dict['user_id'] + Posts_dict['user_id'] 
# function to get unique values
def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

unique_list_of_users = unique(list_of_users)
User_dict = {
    'user_id': [],
    'comment_karma': [],
    'link_karma': [],
    'regitration_date': [],
}


counter = 0
for user_id in unique_list_of_users:
    # if np.isnan(user_id):
    #     continue
    # else:

    if user_id is None:
        continue
    else:
        User_dict['user_id'].append(user_id)
        try:
            User_dict['comment_karma'].append(user_id.comment_karma)
        except:
            User_dict['comment_karma'].append(None)
        try:
            User_dict['link_karma'].append(user_id.link_karma)
        except:
            User_dict['link_karma'].append(None)
        try:
            User_dict['regitration_date'].append(user_id.created_utc)
        except:
            User_dict['regitration_date'].append(None)
    counter = counter + 1
    if counter % 1000 == 0:
        print(str(counter) + ' users info downloaded.')
pd.DataFrame(User_dict).to_csv('users.csv', index=False)

