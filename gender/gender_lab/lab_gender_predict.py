import pandas as pd
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from gender.gender_utils import gender_df_prediction, group_tweets, lemmatization, fit_predict_report
from mining_tweets.retrievedUsers import retrievedLabUsers

lab_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/data/lab_df.pkl').drop_duplicates(
    subset='screen_name', keep='first')
# only users I have tweets of
lab_df = lab_df[lab_df.screen_name.isin(retrievedLabUsers)]

# select random 2500 retweeters from the one I extracted tweets from
sample_lab_users = random.sample(retrievedLabUsers, 2500)

lab_df_sample = lab_df[lab_df.screen_name.isin(sample_lab_users)].loc[:, ['screen_name', 'name', 'description']]
lab_df_sample.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_df_sample.csv')

# select username whose description contains one of the words "woman", "man", "father", "mather".
wo_man = lab_df[lab_df.description.str.contains(r' woman | man | mother | father ', case=False)].loc[:,
         ['screen_name', 'name', 'description']]
wo_man.to_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_wo_man.csv')

# labelling the csv file and reload it - dataset with descriptions
lab_gender = pd.read_csv('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_labelled.csv')
lab_gender.label.value_counts()

# sampling 626 for each category to have a balanced dataset
male = lab_gender[lab_gender.label == 'male'].sample(n=626, random_state=1)
female = lab_gender[lab_gender.label == 'female']

# joining the two samples this does not contain any nan
lab_gender_final = pd.concat([female, male], ignore_index=True)

# creating a dataframe with the joined timeline of the labelled users
path_tweets = '/Users/annalisa/PycharmProjects/MSc_Project/TweetsLab'
grouped_df = group_tweets(lab_gender_final, path_tweets)

pickle.dump(grouped_df,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_grouped_df.pkl', 'wb'),
            protocol=4)

# unpickle grouped fd of tweets
grouped_df = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_grouped_df.pkl')

features = grouped_df.text.values.tolist()  # text column
labels = grouped_df.label.values  # age/label column

# cleaning, lemmatize and tokenize tweets
lemmatized_features = lemmatization(features)
docs_lemma_joined = [" ".join(word) for word in lemmatized_features]

pickle.dump(docs_lemma_joined,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_lemma.pkl', 'wb'),
            protocol=4)

# to be used if the lemmatization has already been performed
docs_lemma_joined = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_lemma.pkl')

'''
similar as lab age
'''
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.9, min_df=0.05, ngram_range=(1, 3))

vectorized_features = vectorizer.fit_transform(docs_lemma_joined).toarray()

# this function can be used to get the name of the vectorized features
# feature_names = vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=0.2, random_state=1)

# Create a Classifier
text_classifier = LogisticRegressionCV(random_state=1, max_iter=300)  # 7529880478087649

fit_predict_report(text_classifier, X_train, y_train, X_test, y_test)

# save the model
pickle.dump(text_classifier,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_model.pkl', 'wb'),
            protocol=4)

# unpickle the model
text_classifier = pickle.load(
    open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_model.pkl', 'rb'))

lab_age_lemma_df_1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_1.pkl')
lab_age_lemma_df_2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_2.pkl')
lab_age_lemma_df_3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_3.pkl')
lab_age_lemma_df_4 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_4.pkl')
lab_age_lemma_df_5 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_5.pkl')
lab_age_lemma_df_6 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_6.pkl')
lab_age_lemma_df_7 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_7.pkl')
lab_age_lemma_df_8 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/age/age_data/lab_age_lemma_df_8.pkl')

lab_gender_predict_1 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_1,
                                            'gender/gender_lab/lab_gender_predict_1.pkl')
lab_gender_predict_2 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_2,
                                            'gender/gender_lab/lab_gender_predict_2.pkl')
lab_gender_predict_3 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_3,
                                            'gender/gender_lab/lab_gender_predict_3.pkl')
lab_gender_predict_4 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_4,
                                            'gender/gender_lab/lab_gender_predict_4.pkl')
lab_gender_predict_5 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_5,
                                            'gender/gender_lab/lab_gender_predict_5.pkl')
lab_gender_predict_6 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_6,
                                            'gender/gender_lab/lab_gender_predict_6.pkl')
lab_gender_predict_7 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_7,
                                            'gender/gender_lab/lab_gender_predict_7.pkl')
lab_gender_predict_8 = gender_df_prediction(text_classifier, vectorizer, lab_age_lemma_df_8,
                                            'gender/gender_lab/lab_gender_predict_8.pkl')
'''
lab_gender_predict_1 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_1.pkl')
lab_gender_predict_2 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_2.pkl')
lab_gender_predict_3 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_3.pkl')
lab_gender_predict_4 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_4.pkl')
lab_gender_predict_5 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_5.pkl')
lab_gender_predict_6 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_6.pkl')
lab_gender_predict_7 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_7.pkl')
lab_gender_predict_8 = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_8.pkl')
'''
# joining all the dataframe of predicitons into one
lab_gender_predicted = pd.concat(
    [lab_gender_predict_1, lab_gender_predict_2, lab_gender_predict_3, lab_gender_predict_4, lab_gender_predict_5,
     lab_gender_predict_6, lab_gender_predict_7, lab_gender_predict_8],
    ignore_index=True)
pickle.dump(lab_gender_predicted,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_ALL.pkl', 'wb'),
            protocol=4)

lab_gender_all = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_ALL.pkl')[
    ['screen_name', 'predicted']]

pickle.dump(lab_gender_all,
            open('/Users/annalisa/PycharmProjects/MSc_Project/gender/gender_lab/lab_gender_predict_ALL_s.pkl', 'wb'),
            protocol=4)

'''
male      26741
female    12953

male      0.673679
female    0.326321
'''


def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=10):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = reversed(sorted(zip(classifier.coef_[labelid], feature_names))[-n:])
    for coef, feat in topn:
        print(classlabel, feat, coef)


most_informative_feature_for_class(vectorizer, text_classifier, 'female')


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()

    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])

    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


show_most_informative_features(vectorizer, text_classifier)
