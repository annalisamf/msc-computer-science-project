import pandas as pd

from mining_tweets import api_call
from mining_tweets import mps_list

# loading the pickle file containing the DataFrame for the Conservative retweeters
con_df = pd.read_pickle('../data/con_df.pkl')

con_users = set([x for x in con_df['screen_name'] if x not in mps_list.mps])

# calling the function to retrieve the tweets on the list of retweeters
api_call.retrieveTweets("../TweetsCon", con_users, "retrievedConUsers")
