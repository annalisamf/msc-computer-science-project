import pandas as pd

from mining_tweets import retrievedUsers

lib_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/data/lib_df.pkl').drop_duplicates(
    subset='screen_name', keep='first')
lib_df = lib_df[lib_df.screen_name.isin(retrievedUsers.retrievedLibUsers)]

# descriptions containing a age-like number
anyAge = lib_df.description.str.match(r'[1-9][0-9]')
lib_age = lib_df[anyAge].loc[:, ['screen_name', 'description']]
lib_age.to_csv('lib_age.csv')

years_old_age = lib_df[lib_df.description.str.match(r'years|yrs|old|born')].loc[:, ['screen_name', 'description']]
years_old_age.to_csv('lib_years_old_age.csv')

wo_man = lib_df[lib_df.description.str.contains(r' woman | man ')].loc[:, ['screen_name', 'description']]
wo_man.to_csv('lib_wo_man.csv')

# descriptions containing the word 'retired'
retired = lib_df[lib_df.description.str.contains('retired')].loc[:, ['screen_name', 'description']].to_csv(
    'lib_retired.csv')

professional = lib_df[lib_df.description.str.contains(r' exper.* | profession | professional ')].loc[:,
               ['screen_name', 'description']]
professional.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/unversioned/age/age_lib/lib_professional.csv')

random_sample = lib_df.loc[:, ['screen_name', 'description']].sample(n=300, random_state=300)
random_sample.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/unversioned/age/age_lib/lib_random.csv')

student = lib_df[lib_df.description.str.contains(r' [Ss]tudent.* | [Uu]niversit.* ')].loc[:,
          ['screen_name', 'description']]
student.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/unversioned/age/age_lib/lib_student.csv')
