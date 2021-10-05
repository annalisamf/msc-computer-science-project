import pandas as pd

from mining_tweets import api_call
from mining_tweets import mps_list

# loading the pickle file containing the DataFrame for the Labours retweeters
lab_df = pd.read_pickle('../data/lab_df.pkl')

lab_users = set([x for x in lab_df['screen_name'] if x not in mps_list.mps])

api_call.retrieveTweets("../TweetsLab", lab_users, "retrievedLabUsers")
