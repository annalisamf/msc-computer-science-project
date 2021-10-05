import pandas as pd

from mining_tweets import retrievedUsers

lab_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/data/lab_df.pkl').drop_duplicates(
    subset='screen_name', keep='first')
lab_df = lab_df[lab_df.screen_name.isin(retrievedUsers.retrievedLabUsers)]

# descriptions containing a age-like number
anyAge = lab_df.description.str.match(r'[1-9][0-9]')
lab_age = lab_df[anyAge].loc[:, ['screen_name', 'description']]
lab_age.to_csv('lab_age.csv')

years_old_age = lab_df[lab_df.description.str.match(r'years|yrs|old|born')].loc[:, ['screen_name', 'description']]
years_old_age.to_csv('lab_years_old_age.csv')

wo_man = lab_df[lab_df.description.str.contains(r' woman | man ')].loc[:, ['screen_name', 'description']]
wo_man.to_csv('lab_wo_man.csv')

# descriptions containing the word 'retired'
retired = lab_df[lab_df.description.str.contains('retired')].loc[:, ['screen_name', 'description']].to_csv(
    'lab_retired.csv')

parent = lab_df[lab_df.description.str.contains(r' father | mother | parent | grand.*')].loc[:,
         ['screen_name', 'description']]
parent.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/unversioned/age/age_lab/lab_parent.csv')
