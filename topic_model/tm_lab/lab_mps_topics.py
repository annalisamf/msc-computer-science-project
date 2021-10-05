from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pyLDAvis.gensim as gensimvis
import pyLDAvis.sklearn
import warnings
from gensim.models.wrappers import LdaMallet
from pprint import pprint

from topic_model.tm_utils import lemmatization, make_bigrams, coh_mallet, mallet_path, words_to_exclude, \
    plot_coh_scores, \
    convertldaMalletToldaGen, term_doc_df, main_topic_doc_df, dict_corpus, freq_topic_df

warnings.filterwarnings('ignore')  # ignore warning message in console

# reading the dataframe with the tweets and joineing all the tweets in one row
df = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_data/lab_mps_tweets.pkl').groupby(
    'screen_name')[
    'text'].apply(lambda x: '. '.join(set(x.dropna()))).reset_index()

# creating a list for each user/timeline
docs = df.text.values.tolist()

# lemmatizing the texts
docs_lemma = lemmatization(docs)

# builing bigrams
docs_bigrams = make_bigrams(docs_lemma)

# save to pickle
pickle.dump(docs_bigrams,
            open('/topic_model/tm_lab/lab_mps_docs_bigrams.pkl', 'wb'),
            protocol=4)
# read to pickle
docs_bigrams = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lab/lab_mps_docs_bigrams.pkl')

# create dictionary and corpus
id2word, corpus = dict_corpus(docs_bigrams, words_to_exclude)

print('Total Vocabulary Size:', len(id2word))

# creating tuples of word frequency in the corpus
word_freq = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# pprint(word_freq)

# create a sorted list of tuple with (word, freq)
words_list = [item for sublist in docs_bigrams for item in sublist]
c = Counter(words_list)
pprint(sorted(c.items(), key=lambda x: x[1], reverse=True)[:10])

'''
# *************** MALLET ***********************
'''
# calculating the umass and cv coherence score for the mallet model from 1 to 20 topics
umass_mallet, cv_mallet = coh_mallet(20, corpus, id2word, docs_bigrams)
pickle.dump(umass_mallet,
            open('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lab/lab_umass_mallet.pkl', 'wb'),
            protocol=4)
pickle.dump(cv_mallet, open('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lab/lab_cv_mallet.pkl', 'wb'),
            protocol=4)

umass_mallet = pd.read_pickle('topic_model/tm_lab/lab_umass_mallet.pkl')
cv_mallet = pd.read_pickle('topic_model/tm_lab/lab_cv_mallet.pkl')

# plotting the coherence scores
plot_coh_scores(range(1, 21), umass_mallet, "umass_coherence Score")
plot_coh_scores(range(1, 21), cv_mallet, "cv_coherence Score")
# mallet model
ldaMallet8 = LdaMallet(mallet_path, corpus=corpus, id2word=id2word, num_topics=8, random_seed=1)
pickle.dump(ldaMallet8, open('topic_model/tm_lab/ldaMallet8.pkl', 'wb'), protocol=4)

ldaMallet8 = pd.read_pickle('topic_model/tm_lab/ldaMallet8.pkl')

pprint(ldaMallet8.print_topics())

# transform the topic model distributions and related corpus data into the data structures needed for the visualization
vis_data = gensimvis.prepare(convertldaMalletToldaGen(ldaMallet8), corpus, id2word, sort_topics=False)
pyLDAvis.save_html(vis_data,
                   '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lab/lab_mps_mallet8.html')

# DISTRIBUTION OF THE TOPICS FOR EACH DOCUMENT IN THE CORPUS

# Term-document dataframe
topics_df = term_doc_df(ldaMallet8)
topics_df.head(10)
topics_df.to_excel('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_lab/lab_topics_df8.xlsx')

# DOMINANT TOPIC FOR EACH DOCUMENT
corpus_topic_df = main_topic_doc_df(ldaMallet8, corpus, df)
corpus_topic_df.head()

# mapping the topic number to a topic name
topic_names = {1: 'Anti-Racism policies', 2: 'Advice surgery',
               3: 'People’s vote', 4: 'Universal credit',
               5: 'Women’s rights',
               6: 'Green industrial', 7: 'Environment', 8: 'Opponent criticism'}

corpus_topic_df['topic_name'] = corpus_topic_df['main_topic'].map(topic_names)
corpus_topic_df.head(20)

# GROUP BY TOPICS
freq_topic_df = freq_topic_df(corpus_topic_df, corpus)
print(freq_topic_df)
pickle.dump(freq_topic_df, open('topic_model/tm_lab/freq_topic_lab.pkl', 'wb'), protocol=4)

freq_topic_df = pd.read_pickle('topic_model/tm_lab/freq_topic_lab.pkl')

# plot the frequency of topics in the corpus
freq_topic_df.sort_values(by=['total_docs'], ascending=False).plot.bar(x='topic_name', y='total_docs', legend=False)
plt.show()

# which document makes the highest contribution to each topic
corpus_topic_df.groupby('main_topic').apply(
    lambda topic_set: (topic_set.sort_values(by=['%_contribution'], ascending=False).iloc[0])).reset_index(drop=True)

# timeline of the user which contribute most to define topic 8
jessphillips = df.loc[df['screen_name'] == 'jessphillips']
pd.set_option('display.max_colwidth', None)
print(jessphillips.text)
