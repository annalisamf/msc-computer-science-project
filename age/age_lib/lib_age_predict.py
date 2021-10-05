from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from age.age_utils import *

# labelling the csv file and reload it - dataset with descriptions
lib_labeled = pd.read_csv('age/age_data/lib_age_labelled.csv', encoding="latin-1",
                          engine='python')  # i left nan so that i could label more
lib_labeled.label.value_counts()

# undersampling
# sampling 132 for each category to have a balanced dataset
sample_30 = lib_labeled[lib_labeled.label == 'below30']
sample_60 = lib_labeled[lib_labeled.label == 'over60'].sample(n=89, random_state=89)
sample_31_59 = lib_labeled[lib_labeled.label == '31-59'].sample(n=89, random_state=89)

# joining the three samples this does not contain any nan
lib_labeled = pd.concat([sample_30, sample_31_59, sample_60], ignore_index=True)

# dataframe with the tweets of the labelled users
path_tweets = '/Users/annalisa/PycharmProjects/MSc_Project/TweetsLib'
grouped_df = group_tweets(lib_labeled, path_tweets)

features = grouped_df.text.values.tolist()  # text column
labels = grouped_df.label.values  # age/label column

# cleaning, lemmatize and tokenize tweets
lemmatized_features = lemmatization(features)
docs_lemma_joined = [" ".join(word) for word in lemmatized_features]

# save lemmatized docs to file
# with open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_docs_lemma.pkl', 'wb') as f:
#     pickle.dump(docs_lemma_joined, f, protocol=4)
pickle.dump(docs_lemma_joined,
            open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_docs_lemma.pkl', 'wb'), protocol=4)

# open pickled file with tokenized tweets
# with open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_docs_lemma.pkl', 'rb') as f:
#     docs_lemma_joined = pickle.load(f)
docs_lemma_joined = pickle.load(
    open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_docs_lemma.pkl', 'rb'))

vectorizer = TfidfVectorizer(max_features=2500, max_df=0.8, min_df=0.05, ngram_range=(1, 2))
vectorized_features = vectorizer.fit_transform(docs_lemma_joined).toarray()
feature_names = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=0.2, random_state=0)

# create a classifier
text_classifier = LogisticRegressionCV(max_iter=300)  # 0.722

# save the model
# need to fit the model on the train data before saving it
# text_classifier.fit(X_train, y_train)
pickle.dump(text_classifier, open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_model.pkl', 'wb'),
            protocol=4)

fit_predict_report(text_classifier, X_train, y_train, X_test, y_test)

most_informative_feature_for_class(vectorizer, text_classifier, 'below30')
most_informative_feature_for_class(vectorizer, text_classifier, '31-59')
most_informative_feature_for_class(vectorizer, text_classifier, 'over60')

# PREDICTING AGE OF USERS
from mining_tweets import retrievedUsers

lib_retweeters = retrievedUsers.retrievedLibUsers
lib_age_model = pickle.load(open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_model.pkl', 'rb'))

# FIRST SET
# dataframe with lemmatized timeline
lib_age_lemma_df_1 = df_lemmatized(lib_retweeters[:5000], path_tweets, 'lib_age_lemma_df_1.pkl')
# datarame with predicted age
lib_age_predict_1 = age_df_prediction(lib_age_model, vectorizer, lib_age_lemma_df_1, 'lib_age_predict_1.pkl')

# SECOND SET
# dataframe with lemmatized timeline
lib_age_lemma_df_2 = df_lemmatized(lib_retweeters[5000:10000], path_tweets, 'lib_age_lemma_df_2.pkl')
# datarame with predicted age
lib_age_predict_2 = age_df_prediction(lib_age_model, vectorizer, lib_age_lemma_df_2, 'lib_age_predict_2.pkl')

# THIRD SET
# dataframe with lemmatized timeline
lib_age_lemma_df_3 = df_lemmatized(lib_retweeters[10000:], path_tweets, 'lib_age_lemma_df_3.pkl')
# datarame with predicted age
lib_age_predict_3 = age_df_prediction(lib_age_model, vectorizer, lib_age_lemma_df_3, 'lib_age_predict_3.pkl')


# build a unique dataframe with all the age prediction for labour retweeters
lib_age1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_predict_1.pkl')[
    ['screen_name', 'predicted']]
lib_age2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_predict_2.pkl')[
    ['screen_name', 'predicted']]
lib_age3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lib_age_predict_3.pkl')[
    ['screen_name', 'predicted']]

lib_age_predicted = pd.concat([lib_age1, lib_age2, lib_age3],
                              ignore_index=True)
pickle.dump(lib_age_predicted,
            open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_lib/lib_age_predicted.pkl', 'wb'), protocol=4)

# load df with all age predictions
lib_age_all = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_lib/lib_age_predicted.pkl')[
    ['screen_name', 'predicted']]

'''
31-59      12442
over60       928
below30      636

31-59      0.888334
over60     0.066257
below30    0.045409
'''