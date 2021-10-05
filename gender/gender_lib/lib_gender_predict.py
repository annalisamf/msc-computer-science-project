import pandas as pd
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from gender.gender_utils import gender_df_prediction, group_tweets, lemmatization, fit_predict_report
from mining_tweets.retrievedUsers import retrievedLibUsers

lib_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/data/lib_df.pkl').drop_duplicates(
    subset='screen_name', keep='first')
# only users I have tweets of
lib_df = lib_df[lib_df.screen_name.isin(retrievedLibUsers)]

# select random 2500 retweeters from the one I extracted tweets from
sample_lib_users = random.sample(retrievedLibUsers, 2500)

lib_df_sample = lib_df[lib_df.screen_name.isin(sample_lib_users)].loc[:, ['screen_name', 'name', 'description']]
lib_df_sample.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_df_sample.csv')

# select username whose description contains one of the words "woman", "man", "father", "mather".
wo_man = lib_df[lib_df.description.str.contains(r' woman | man | mother | father ', case=False)].loc[:,
         ['screen_name', 'name', 'description']]
wo_man.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_wo_man.csv')

# labelling the csv file and reload it - dataset with descriptions
lib_gender = pd.read_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_labelled.csv')
lib_gender.label.value_counts()

# sampling 424 for each category to have a balanced dataset
male = lib_gender[lib_gender.label == 'male'].sample(n=424, random_state=1)
female = lib_gender[lib_gender.label == 'female']

# joining the two samples this does not contain any nan
lib_gender_final = pd.concat([female, male], ignore_index=True)

# creating a dataframe with the joined timeline of the labelled users
path_tweets = '/Users/annalisa/PycharmProjects/MSc_Project/TweetsLib'
grouped_df = group_tweets(lib_gender_final, path_tweets)

pickle.dump(grouped_df,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_grouped_df.pkl', 'wb'),
            protocol=4)

# unpickle grouped fd of tweets
grouped_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_grouped_df.pkl')

features = grouped_df.text.values.tolist()  # text column
labels = grouped_df.label.values  # age/label column

# cleaning, lemmatize and tokenize tweets
lemmatized_features = lemmatization(features)
docs_lemma_joined = [" ".join(word) for word in lemmatized_features]

pickle.dump(docs_lemma_joined,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_lemma.pkl', 'wb'),
            protocol=4)

docs_lemma_joined = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_lemma.pkl')

'''
similar to lib age
'''

# vectorizer = TfidfVectorizer(max_features=10000, max_df=0.8, min_df=0.05, ngram_range=(1, 3)) #ok
vectorizer = TfidfVectorizer(max_features=10000, max_df=0.9, min_df=0.05, ngram_range=(1, 3))
vectorized_features = vectorizer.fit_transform(docs_lemma_joined).toarray()

# this function can be used to get the name of the vectorized features
# feature_names = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=0.2, random_state=1)

# Create a Classifier
# text_classifier = LogisticRegressionCV(cv=4, n_jobs=100, max_iter=1000, random_state=1) #ok
text_classifier = LogisticRegressionCV(max_iter=300, random_state=1)

fit_predict_report(text_classifier, X_train, y_train, X_test, y_test)

# save the model
pickle.dump(text_classifier,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_model.pkl', 'wb'),
            protocol=4)

# unpickle the model
text_classifier = pickle.load(
    open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_model.pkl', 'rb'))

lib_age_lemma_df_1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_lemma_df_1.pkl')
lib_age_lemma_df_2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_lemma_df_2.pkl')
lib_age_lemma_df_3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_lemma_df_3.pkl')

lib_gender_predict_1 = gender_df_prediction(text_classifier, vectorizer, lib_age_lemma_df_1,
                                            'gender/gender_lib/lib_gender_predict_1.pkl')
lib_gender_predict_2 = gender_df_prediction(text_classifier, vectorizer, lib_age_lemma_df_2,
                                            'gender/gender_lib/lib_gender_predict_2.pkl')
lib_gender_predict_3 = gender_df_prediction(text_classifier, vectorizer, lib_age_lemma_df_3,
                                            'gender/gender_lib/lib_gender_predict_3.pkl')
'''
lib_gender_predict_1 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_predict_1.pkl')
lib_gender_predict_2 = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_predict_2.pkl')
'''

# joining all the dataframe of predicitons into one
lib_gender_predicted = pd.concat([lib_gender_predict_1, lib_gender_predict_2, lib_gender_predict_3],
                                 ignore_index=True)
pickle.dump(lib_gender_predicted,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_predict_ALL.pkl', 'wb'),
            protocol=4)

lib_gender_all = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_predict_ALL.pkl')[
    ['screen_name', 'predicted']]

pickle.dump(lib_gender_all,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lib/lib_gender_predict_ALL_s.pkl', 'wb'),
            protocol=4)

'''
male      8383
female    5623

male      0.598529
female    0.401471
'''
