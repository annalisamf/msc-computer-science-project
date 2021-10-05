import pandas as pd

from mining_tweets import retrievedUsers

con_df = pd.read_pickle('data/con_df.pkl').drop_duplicates(
    subset='screen_name', keep='first')
con_df = con_df[con_df.screen_name.isin(retrievedUsers.retrievedConUsers)]

# descriptions containing a age-like number
anyAge = con_df.description.str.match(r'[1-9][0-9]')
# descriptions containing the word 'retired'
retired = con_df[con_df.description.str.contains('retired')].loc[:, ['description']]
con_age = con_df[anyAge].loc[:, ['screen_name', 'description']]
# con_age.to_csv('con_age_labelled.csv')

years_old_age = con_df[con_df.description.str.match(r'years|yrs|old|born')].loc[:, ['screen_name', 'description']]
years_old_age.to_csv('con_years_old.csv')

wo_man = con_df[con_df.description.str.contains(r' woman | man ')].loc[:, ['screen_name', 'description']]
wo_man.to_csv('con_wo_man.csv')

student = con_df[con_df.description.str.contains(r' stud.*')].loc[:, ['screen_name', 'description']]
student.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_student.csv')

parent = con_df[con_df.description.str.contains(r' father | mother | parent ')].loc[:, ['screen_name', 'description']]
parent.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_parent.csv')

professional = con_df[con_df.description.str.contains(r' exper.* | profession | professional ')].loc[:,
               ['screen_name', 'description']]
professional.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_professional.csv')
