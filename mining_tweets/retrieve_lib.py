import pandas as pd

from mining_tweets import api_call
from mining_tweets import mps_list

# loading the pickle file containing the DataFrame for the LibDem retweeters
lib_df = pd.read_pickle('../data/lib_df.pkl')

lib_users = set([x for x in lib_df['screen_name'] if x not in mps_list.mps])

api_call.retrieveTweets("../TweetsLib", lib_users, "retrievedLibUsers")
