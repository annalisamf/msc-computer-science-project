import pandas as pd
import pickle
import spacy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
spacy.load('en_vectors_web_lg', vocab=nlp.vocab)
nlp.max_length = 1500000
stopwords = nlp.Defaults.stop_words | {'re', 'rt', 'co', 'amp'}


def gender_df_prediction(model, vectorizer, lemma_df, save_predict_df):
    vectorized_features = vectorizer.fit_transform(lemma_df.text).toarray()
    predict = model.predict(vectorized_features)
    lemma_df['predicted'] = predict
    pickle.dump(lemma_df, open('/Users/annalisa/PycharmProjects/MSc_Project'
                               '/%s' % save_predict_df, 'wb'), protocol=4)
    return lemma_df


def lemmatization(list_of_docs):  # returns lemma only of nouns, adj, verbs and adv
    keep_tags = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']
    lemmatized_text = []
    for sent in list_of_docs:
        # doc = nlp(" ".join(sent))  # we need to pass a string of words, not a list
        doc = nlp(sent)  # we need to pass a string of words, not a list
        lemmatized_text.append(
            [token.lemma_.lower() for token in doc if
             token.pos_ in keep_tags and token.is_alpha and len(
                 token.lemma_) > 1 and token.lemma_.strip().lower() not in stopwords])
        #     is_apha removes number, punctuation and urls
        print(f"{len(lemmatized_text)} sentences lemmatized")
        # print(lemmatized_text[0])
    return lemmatized_text


# this function creates a dataset with the tweets of a selected list of users
def joined_df(list_of_users, path_tweets):
    li = []
    for user in list_of_users:
        li.append(pd.read_csv(path_tweets + "/%s_tweets.csv" % user,
                              index_col=None, header=0))
    return pd.concat(li, axis=0, ignore_index=True)


# this function joins timelines of single users and selects only name and text
def group_tweets(labelled_df, path_tweets):
    # create dataframe with tweets of labelled users(joined) and group the timeline in one row per user
    grouped_df = joined_df(labelled_df.screen_name.tolist(), path_tweets).groupby('screen_name')['text'].apply(
        lambda x: '. '.join(x)).reset_index()
    # merging the names/labels with the tweets, and keeping only relevant columns (text and labels)
    grouped_df = pd.merge(grouped_df, labelled_df, on='screen_name')[['text', 'label']]
    return grouped_df


# fit the model and print the score report
def fit_predict_report(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, test_predict))
    print("Classification Report")
    print(classification_report(y_test, test_predict))
    print(f'Accuracy on training set {accuracy_score(y_train, train_predict)}')
    print(f'Accuracy on test set {accuracy_score(y_test, test_predict)}')
