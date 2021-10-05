from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from age.age_utils import *

# labelling the csv file and reload it - dataset with descriptions
con_labeled = pd.read_csv(
    '/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_labelled.csv')  # I left nan so that I could label more in the future
con_labeled.label.value_counts()

# sampling 267 for each category to have a balanced dataset
sample_30 = con_labeled[con_labeled.label == 'below30'].sample(n=267, random_state=1)
sample_60 = con_labeled[con_labeled.label == 'over60'].sample(n=267, random_state=1)
sample_31_59 = con_labeled[con_labeled.label == '31-59']

# joining the three samples this does not contain any nan
con_labeled = pd.concat([sample_30, sample_31_59, sample_60], ignore_index=True)

# creating a dataframe with the joined timeline of the labelled users
path_tweets = '/Users/annalisa/PycharmProjects/MSc_Project/TweetsCon'
grouped_df = group_tweets(con_labeled, path_tweets)

features = grouped_df.text.values.tolist()  # text column
labels = grouped_df.label.values  # age/label column

# cleaning, lemmatize and tokenize tweets
lemmatized_features = lemmatization(features)
docs_lemma_joined = [" ".join(word) for word in lemmatized_features]

# save lemmatized docs to file
# with open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_docs_lemma.pkl', 'wb') as f:
#     pickle.dump(docs_lemma_joined, f, protocol=4)
pickle.dump(docs_lemma_joined,
            open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_docs_lemma.pkl', 'wb'), protocol=4)

# open pickled file with tokenized tweets
# with open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_docs_lemma.pkl', 'rb') as f:
#     docs_lemma_joined = pickle.load(f)
docs_lemma_joined = pickle.load(
    open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_docs_lemma.pkl', 'rb'))

vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7, min_df=0.05, ngram_range=(1, 2))  # 7080
vectorized_features = vectorizer.fit_transform(docs_lemma_joined).toarray()
feature_names = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=0.2, random_state=1)

text_classifier = LogisticRegressionCV(max_iter=300)  # 0.7080745341614907, default cv is 5

fit_predict_report(text_classifier, X_train, y_train, X_test, y_test)

# save the model
# need to fit the model on the train data before saving it
pickle.dump(text_classifier, open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_model.pkl', 'wb'),
            protocol=4)

show_most_informative_features(vectorizer, text_classifier, n=10)

# testing the model
from mining_tweets import retrievedUsers

con_retweeters = retrievedUsers.retrievedConUsers
# load the model
con_age_model = pickle.load(open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_model.pkl', 'rb'))

# FIRST SET
# dataframe with lemmatized timeline
con_age_lemma_df_1 = df_lemmatized(con_retweeters[:5000], path_tweets, 'con_age_lemma_df_1.pkl')
# dataframe with predicted age
con_age_predict_1 = age_df_prediction(con_age_model, vectorizer, con_age_lemma_df_1, 'con_age_predict_1.pkl')

# SECOND SET
# dataframe with lemmatized timeline
con_age_lemma_df_2 = df_lemmatized(con_retweeters[5000:10000], path_tweets, 'con_age_lemma_df_2.pkl')
# dataframe with predicted age
con_age_predict_2 = age_df_prediction(con_age_model, vectorizer, con_age_lemma_df_2, 'con_age_predict_2.pkl')

# THIRD SET
# dataframe with lemmatized timeline
con_age_lemma_df_3 = df_lemmatized(con_retweeters[10000:15000], path_tweets, 'con_age_lemma_df_3.pkl')
# dataframe with predicted age
con_age_predict_3 = age_df_prediction(con_age_model, vectorizer, con_age_lemma_df_3, 'con_age_predict_3.pkl')

# FOURTH SET
# dataframe with lemmatized timeline
con_age_lemma_df_4 = df_lemmatized(con_retweeters[15000:20000], path_tweets, 'con_age_lemma_df_4.pkl')
# dataframe with predicted age
con_age_predict_4 = age_df_prediction(con_age_model, vectorizer, con_age_lemma_df_4, 'con_age_predict_4.pkl')

# FIFTH SET
# dataframe with lemmatized timeline
con_age_lemma_df_5 = df_lemmatized(con_retweeters[20000:25000], path_tweets, 'con_age_lemma_df_5.pkl')
# dataframe with predicted age
con_age_predict_5 = age_df_prediction(con_age_model, vectorizer, con_age_lemma_df_5, 'con_age_predict_5.pkl')

# SIXTH SET
# dataframe with lemmatized timeline
con_age_lemma_df_6 = df_lemmatized(con_retweeters[25000:], path_tweets, 'con_age_lemma_df_6.pkl')
# dataframe with predicted age
con_age_predict_6 = age_df_prediction(con_age_model, vectorizer, con_age_lemma_df_6, 'con_age_predict_6.pkl')

# build a unique dataframe with all the age prediction for conservatives retweeters
con_age1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_predict_1.pkl')[
    ['screen_name', 'predicted']]
con_age2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_predict_2.pkl')[
    ['screen_name', 'predicted']]
con_age3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_predict_3.pkl')[
    ['screen_name', 'predicted']]
con_age4 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_predict_4.pkl')[
    ['screen_name', 'predicted']]
con_age5 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_predict_5.pkl')[
    ['screen_name', 'predicted']]
con_age6 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/con_age_predict_6.pkl')[
    ['screen_name', 'predicted']]

con_age_predicted = pd.concat([con_age1, con_age2, con_age3, con_age4, con_age5, con_age6], ignore_index=True)
pickle.dump(con_age_predicted,
            open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_con/con_age_predicted.pkl', 'wb'), protocol=4)

con_age_all = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_con/con_age_predicted.pkl')[
    ['screen_name', 'predicted']]

'''
31-59      20837
over60      6021
below30     1987

31-59      0.722378
over60     0.208736
below30    0.068885

'''
