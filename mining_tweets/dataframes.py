import json

import pandas as pd


# the DataFrames are slow to create so I create them once and store them in an external file (pickle)

# function to create the DataFrames from the retweeters file (eg. con_retweeters),
# with name of retweeters and description
def create_df(file, output):
    df = pd.read_csv(file, header=None, delimiter=r"\t", engine='python')
    df.columns = ["tweet_ID", "user_ID", "Metadata"]

    df = df.join(df['Metadata'].apply(json.loads).apply(pd.Series))
    df = df[['tweet_ID', 'user_ID', 'screen_name', 'name', 'description', 'location',
             'geo_enabled', 'followers_count', 'favourites_count', 'friends_count', 'id_str']]
    df.to_pickle('%s.pkl' % output, protocol=4)


# creating a DataFrames and storing in a pkl file for each of the retweeters file
create_df("../data/con_retweeters.csv", 'con_df')
create_df("../data/lab_retweeters.csv", 'lab_df')
create_df("../data/lib_retweeters.csv", 'lib_df')
