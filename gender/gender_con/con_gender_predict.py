import pandas as pd
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from gender.gender_utils import gender_df_prediction, group_tweets, lemmatization, fit_predict_report
from mining_tweets.retrievedUsers import retrievedConUsers

con_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/data/con_df.pkl').drop_duplicates(
    subset='screen_name', keep='first')
# only users I have tweets of
con_df = con_df[con_df.screen_name.isin(retrievedConUsers)]

# select random 2500 retweeters from the one I extracted tweets from
sample_con_users = random.sample(retrievedConUsers, 2500)

con_df_sample = con_df[con_df.screen_name.isin(sample_con_users)].loc[:, ['screen_name', 'name', 'description']]
# saving to excel
# con_df_sample.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_df_sample.csv')

wo_man = con_df[con_df.description.str.contains(r' woman | man | mother | father ', case=False)].loc[:,
         ['screen_name', 'name', 'description']]
# saving to excel
# wo_man.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_wo_man.csv')

# labelling the csv file and reload it - dataset with descriptions
con_gender = pd.read_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_labelled.csv')
con_gender.label.value_counts()
con_gender = con_gender[con_gender.screen_name.isin(retrievedConUsers)]

# sampling 422 for each category to have a balanced dataset
male = con_gender[con_gender.label == 'male'].sample(n=419, random_state=1)
female = con_gender[con_gender.label == 'female']

# joining the two samples this does not contain any nan
con_gender_final = pd.concat([female, male], ignore_index=True)

# creating a dataframe with the joined timeline of the labelled users
path_tweets = '/Users/annalisa/PycharmProjects/MSc_Project/TweetsCon'
grouped_df = group_tweets(con_gender_final, path_tweets)

pickle.dump(grouped_df,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_grouped_df.pkl', 'wb'),
            protocol=4)

# unpickle grouped df of tweets
grouped_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_grouped_df.pkl')

features = grouped_df.text.values.tolist()  # text column
labels = grouped_df.label.values  # age/label column

# cleaning, lemmatize and tokenize tweets
lemmatized_features = lemmatization(features)
docs_lemma_joined = [" ".join(word) for word in lemmatized_features]

pickle.dump(docs_lemma_joined,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_lemma.pkl', 'wb'),
            protocol=4)

# to be used if the lemmatization has already been performed
docs_lemma_joined = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_lemma.pkl')

'''
identical to age
'''
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.7, min_df=0.05, ngram_range=(1, 3))
vectorized_features = vectorizer.fit_transform(docs_lemma_joined).toarray()

# this function can be used to get the name of the vectorized features
# feature_names = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=0.2, random_state=1)

text_classifier = LogisticRegressionCV(random_state=1, max_iter=300)

fit_predict_report(text_classifier, X_train, y_train, X_test, y_test)

# save the model
pickle.dump(text_classifier,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_model.pkl', 'wb'),
            protocol=4)

# unpickle the model
text_classifier = pickle.load(
    open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_model.pkl', 'rb'))

# using the same lemmatized tweetsused for the age
con_age_lemma_df_1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_lemma_df_1.pkl')
con_age_lemma_df_2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_lemma_df_2.pkl')
con_age_lemma_df_3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_lemma_df_3.pkl')
con_age_lemma_df_4 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_lemma_df_4.pkl')
con_age_lemma_df_5 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_lemma_df_5.pkl')
con_age_lemma_df_6 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_lemma_df_6.pkl')

con_gender_predict_1 = gender_df_prediction(text_classifier, vectorizer, con_age_lemma_df_1,
                                            'gender/gender_con/con_gender_predict_1.pkl')
con_gender_predict_2 = gender_df_prediction(text_classifier, vectorizer, con_age_lemma_df_2,
                                            'gender/gender_con/con_gender_predict_2.pkl')
con_gender_predict_3 = gender_df_prediction(text_classifier, vectorizer, con_age_lemma_df_3,
                                            'gender/gender_con/con_gender_predict_3.pkl')
con_gender_predict_4 = gender_df_prediction(text_classifier, vectorizer, con_age_lemma_df_4,
                                            'gender/gender_con/con_gender_predict_4.pkl')
con_gender_predict_5 = gender_df_prediction(text_classifier, vectorizer, con_age_lemma_df_5,
                                            'gender/gender_con/con_gender_predict_5.pkl')
con_gender_predict_6 = gender_df_prediction(text_classifier, vectorizer, con_age_lemma_df_6,
                                            'gender/gender_con/con_gender_predict_6.pkl')

con_gender_predict_1 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_1.pkl')
con_gender_predict_2 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_2.pkl')
con_gender_predict_3 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_3.pkl')
con_gender_predict_4 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_4.pkl')
con_gender_predict_5 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_5.pkl')
con_gender_predict_6 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_6.pkl')

con_gender_predict_all = pd.concat(
    [con_gender_predict_1, con_gender_predict_2, con_gender_predict_3, con_gender_predict_4, con_gender_predict_5,
     con_gender_predict_6],
    ignore_index=True)
pickle.dump(con_gender_predict_all,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_ALL.pkl', 'wb'),
            protocol=4)

con_gender_all = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_ALL_s.pkl')[
    ['screen_name', 'predicted']]
pickle.dump(con_gender_all,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_con/con_gender_predict_ALL_s.pkl', 'wb'),
            protocol=4)

'''
male      17078
female    11767

male      0.592061
female    0.407939

'''
