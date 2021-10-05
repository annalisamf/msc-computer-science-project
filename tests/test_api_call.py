import pandas as pd

from mining_tweets.api_call import *


#  testing the api call and retrieving the tweets from Birkbeck account
#  testing that the file is saved in the data folder and that the screen name correspond to BirkbeckUoL
def test_get_tweets():
    screen_name = 'BirkbeckUoL'
    folder = '/Users/annalisa/PycharmProjects/MSc_Project/tests/data'
    assert get_tweets(folder, screen_name) == None
    df_tweets = pd.read_csv('/Users/annalisa/PycharmProjects/MSc_Project/tests/data/BirkbeckUoL_tweets.csv')
    assert df_tweets.screen_name[0] == 'BirkbeckUoL'
