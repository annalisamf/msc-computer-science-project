# creates dataframes with mps/timeline rows
import glob
import json
import os
import pickle
import re

import pandas as pd

path_con = '//Users/annalisa/PycharmProjects/MSc_Project/data/con'
path_lab = '/Users/annalisa/PycharmProjects/MSc_Project/data/lab'
path_lib = '/Users/annalisa/PycharmProjects/MSc_Project/data/lib'


# Extrating the usernames from the files
# this function creates a list of names of files. Each csv is named with the candidate name
def names_from_file(path):
    return list(map(lambda x: x.replace('_2019-12-06_tweets.txt', ''), os.listdir(path)))


con_names_from_file = names_from_file(path_con)
lab_names_from_file = names_from_file(path_lab)
lib_names_from_file = names_from_file(path_lib)


# creates a dataframe with joined tweets of the mps
def create_joined_df(path, save_to):
    all_files = glob.glob(path + "/*.txt")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, delimiter=r"\t", engine='python')
        df['screen_name'] = re.sub('_2019.*', '', filename).replace(f'{path}/', '')
        li.append(df)
        print(len(li))
    joined_frame = pd.concat(li, axis=0, ignore_index=True)
    joined_frame.to_pickle(save_to, protocol=4)


create_joined_df(path_con, '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_data/con_mps_tweets.pkl')
create_joined_df(path_lab, '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_data/lab_mps_tweets.pkl')
create_joined_df(path_lib, '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_data/lib_mps_tweets.pkl')
