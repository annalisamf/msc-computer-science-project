from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from age.age_utils import *

# labelling the csv file and reload it - dataset with descriptions
lab_labeled = pd.read_csv('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_labelled.csv',
                          encoding="latin-1", engine='python')  # i left nan so that i could label more
lab_labeled.label.value_counts()

# undersampling
# sampling 132 for each category to have a balanced dataset
sample_30 = lab_labeled[lab_labeled.label == 'below30'].sample(n=190, random_state=195)
sample_60 = lab_labeled[lab_labeled.label == 'over60'].sample(n=190, random_state=195)
sample_31_59 = lab_labeled[lab_labeled.label == '31-59']

# joining the three samples this does not contain any nan
lab_labeled = pd.concat([sample_30, sample_31_59, sample_60], ignore_index=True)

# dataframe with the tweets of the labelled users
path_tweets = '/Users/annalisa/PycharmProjects/MSc_Project/Tweetslab'
grouped_df = group_tweets(lab_labeled, path_tweets)

features = grouped_df.text.values.tolist()  # text column
labels = grouped_df.label.values  # age/label column

# cleaning, lemmatize and tokenize tweets
lemmatized_features = lemmatization(features)
docs_lemma_joined = [" ".join(word) for word in lemmatized_features]

# save lemmatized docs to file
# with open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_docs_lemma.pkl', 'wb') as f:
#     pickle.dump(docs_lemma_joined, f, protocol=4)
pickle.dump(docs_lemma_joined,
            open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_docs_lemma.pkl', 'wb'), protocol=4)

# open pickled file with tokenized tweets
# with open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_docs_lemma.pkl', 'rb') as f:
#     docs_lemma_joined = pickle.load(f)
docs_lemma_joined = pickle.load(
    open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_docs_lemma.pkl', 'rb'))

vectorizer = TfidfVectorizer(max_features=5000, max_df=0.7, min_df=0.05, ngram_range=(1, 3))
# vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True, ngram_range=(1, 2))
vectorized_features = vectorizer.fit_transform(docs_lemma_joined).toarray()

feature_names = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=0.2, random_state=0)

# Create a Classifier
text_classifier = LogisticRegressionCV(cv=4, max_iter=1000, n_jobs=100)  # 7192982456140351

fit_predict_report(text_classifier, X_train, y_train, X_test, y_test)

# save the model
# need to fit the model on the train data before saving it
pickle.dump(text_classifier, open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_model.pkl', 'wb'),
            protocol=4)

most_informative_feature_for_class(vectorizer, text_classifier, 'below30')
most_informative_feature_for_class(vectorizer, text_classifier, '31-59')
most_informative_feature_for_class(vectorizer, text_classifier, 'over60')

# testing
from mining_tweets import retrievedUsers

lab_retweeters = retrievedUsers.retrievedLabUsers
lab_age_model = pickle.load(open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_model.pkl', 'rb'))

# FIRST SET
# dataframe with lemmatized timeline
lab_age_lemma_df_1 = df_lemmatized(lab_retweeters[:5000], path_tweets, 'lab_age_lemma_df_1.pkl')
# datarame with predicted age
lab_age_predict_1 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_1, 'lab_age_predict_1.pkl')

# SECOND SET
# dataframe with lemmatized timeline
lab_age_lemma_df_2 = df_lemmatized(lab_retweeters[5000:10000], path_tweets, 'lab_age_lemma_df_2.pkl')
# datarame with predicted age
lab_age_predict_2 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_2, 'lab_age_predict_2.pkl')

# THIRD SET
# dataframe with lemmatized timeline
lab_age_lemma_df_3 = df_lemmatized(lab_retweeters[10000:15000], path_tweets, 'lab_age_lemma_df_3.pkl')
# datarame with predicted age
lab_age_predict_3 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_3, 'lab_age_predict_3.pkl')

# FOURTH SET
# dataframe with lemmatized timeline
lab_age_lemma_df_4 = df_lemmatized(lab_retweeters[15000:20000], path_tweets, 'lab_age_lemma_df_4.pkl')
# datarame with predicted age
lab_age_predict_4 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_4, 'lab_age_predict_4.pkl')

# FIFTH SET
# dataframe with lemmatized timeline
lab_age_lemma_df_5 = df_lemmatized(lab_retweeters[20000:25000], path_tweets, 'lab_age_lemma_df_5.pkl')
# datarame with predicted age
lab_age_predict_5 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_5, 'lab_age_predict_5.pkl')

# SIXTH SET
# dataframe with lemmatized timeline
lab_age_lemma_df_6 = df_lemmatized(lab_retweeters[25000:30000], path_tweets, 'lab_age_lemma_df_6.pkl')
# datarame with predicted age
lab_age_predict_6 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_6, 'lab_age_predict_6.pkl')

# SEVENTH SET
# dataframe with lemmatized timeline
lab_age_lemma_df_7 = df_lemmatized(lab_retweeters[30000:35000], path_tweets, 'lab_age_lemma_df_7.pkl')
# datarame with predicted age
lab_age_predict_7 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_7, 'lab_age_predict_7.pkl')

# EIGHTH SET
# dataframe with lemmatized timeline
lab_age_lemma_df_8 = df_lemmatized(lab_retweeters[35000:], path_tweets, 'lab_age_lemma_df_8.pkl')
# datarame with predicted age
lab_age_predict_8 = age_df_prediction(lab_age_model, vectorizer, lab_age_lemma_df_8, 'lab_age_predict_8.pkl')

# build a unique dataframe with all the age prediction for labour retweeters
lab_age1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_1.pkl')[
    ['screen_name', 'predicted']]
lab_age2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_2.pkl')[
    ['screen_name', 'predicted']]
lab_age3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_3.pkl')[
    ['screen_name', 'predicted']]
lab_age4 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_4.pkl')[
    ['screen_name', 'predicted']]
lab_age5 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_5.pkl')[
    ['screen_name', 'predicted']]
lab_age6 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_6.pkl')[
    ['screen_name', 'predicted']]
lab_age7 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_7.pkl')[
    ['screen_name', 'predicted']]
lab_age8 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_predict_8.pkl')[
    ['screen_name', 'predicted']]

lab_age_predicted = pd.concat([lab_age1, lab_age2, lab_age3, lab_age4, lab_age5, lab_age6, lab_age7, lab_age8],
                              ignore_index=True)
pickle.dump(lab_age_predicted,
            open('/Users/annalisa/PycharmProjects/MSc_Project/age/age_lab/lab_age_predicted.pkl', 'wb'), protocol=4)

lab_age_all = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_lab/lab_age_predicted.pkl')[
    ['screen_name', 'predicted']]

'''
31-59      36087
over60      2697
below30      910

31-59      0.909130
over60     0.067945
below30    0.022925
'''
