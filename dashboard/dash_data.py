import pickle

import pandas as pd

############ DATA ##########
### AGE DATA
con_age_predicted = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_con/con_age_predicted.pkl')[
    ['screen_name', 'predicted']]
lab_age_predicted = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_lab/lab_age_predicted.pkl')[
    ['screen_name', 'predicted']]
lib_age_predicted = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_lib/lib_age_predicted.pkl')[
    ['screen_name', 'predicted']]

con_age_predicted['party'] = 'Conservative'
lab_age_predicted['party'] = 'Labour'
lib_age_predicted['party'] = 'LibDem'

age = pd.concat([con_age_predicted, lab_age_predicted, lib_age_predicted])

pickle.dump(age, open('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/age.pkl', 'wb'), protocol=4)

### GENDER DATA
con_gender_predicted = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_ALL_s.pkl')
lab_gender_predicted = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_ALL_s.pkl')
lib_gender_predicted = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_predict_ALL_s.pkl')

con_gender_predicted['party'] = 'Conservative'
lab_gender_predicted['party'] = 'Labour'
lib_gender_predicted['party'] = 'LibDem'

gender = pd.concat([con_gender_predicted, lab_gender_predicted, lib_gender_predicted])

pickle.dump(gender, open('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/gender.pkl', 'wb'), protocol=4)

### LOCATION DATA
con_locations = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/location/con_locations.pkl')
lab_locations = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/location/lab_locations.pkl')
lib_locations = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/location/lib_locations.pkl')

con_locations['party'] = 'Conservative'
lab_locations['party'] = 'Labour'
lib_locations['party'] = 'LibDem'

location = pd.concat([con_locations, lab_locations, lib_locations])

pickle.dump(location, open('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/location.pkl', 'wb'), protocol=4)

### TOPICS DATA
con_topics = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_con/freq_topic_con.pkl')
lab_topics = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lab/freq_topic_lab.pkl')
lib_topics = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lib/freq_topic_lib.pkl')

con_topics['party'] = 'Conservative'
lab_topics['party'] = 'Labour'
lib_topics['party'] = 'LibDem'

topics = pd.concat([con_topics, lab_topics, lib_topics])

pickle.dump(topics, open('/Users/annalisa/PycharmProjects/MSc_Project/dashboard/topics.pkl', 'wb'), protocol=4)
