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

# reading the dataframe with the tweets and joining all the tweets in one row
tweets_df = pd.read_pickle(
    '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_data/con_mps_tweets.pkl').groupby(
    'screen_name')[
    'text'].apply(lambda x: '. '.join(set(x.dropna()))).reset_index()

# creating a list for each user/timeline
docs = tweets_df.text.values.tolist()

# lemmatizing the texts
docs_lemma = lemmatization(docs)

# builing bigrams
docs_bigrams = make_bigrams(docs_lemma)

# save to pickle
pickle.dump(docs_bigrams,
            open('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_con/con_mps_docs_bigrams.pkl', 'wb'),
            protocol=4)
# read to pickle
docs_bigrams = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_con/con_mps_docs_bigrams.pkl')

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

pickle.dump(umass_mallet, open('topic_model/tm_con/con_umass_mallet.pkl', 'wb'), protocol=4)
pickle.dump(cv_mallet, open('topic_model/tm_con/con_cv_mallet.pkl', 'wb'), protocol=4)

umass_mallet = pd.read_pickle('topic_model/tm_con/con_umass_mallet.pkl')
cv_mallet = pd.read_pickle('topic_model/tm_con/con_cv_mallet.pkl')

plot_coh_scores(range(1, 21), umass_mallet, "umass_coherence Score")
plot_coh_scores(range(1, 21), cv_mallet, "cv_coherence Score")

ldaMallet9 = LdaMallet(mallet_path, corpus=corpus, id2word=id2word, num_topics=9, random_seed=1)

# save ldamallet as pkl
# pickle.dump(ldaMallet9, open('topic_model/tm_con/ldaMallet9.pkl', 'wb'), protocol=4)
# load ldamallet
# ldaMallet9 = pickle.load(open('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_con/ldaMallet9.pkl', 'rb'))

pprint(ldaMallet9.print_topics())

# transform the topic model distributions and related corpus data into the data structures needed for the visualization
vis_data = gensimvis.prepare(convertldaMalletToldaGen(ldaMallet9), corpus, id2word, sort_topics=False)
pyLDAvis.save_html(vis_data,
                   '/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_con/con_mps_mallet9'
                   '.html')

# DISTRIBUTION OF THE TOPICS FOR EACH DOCUMENT IN THE CORPUS

# Term-document dataframe
topics_df = term_doc_df(ldaMallet9)
topics_df.head(10)
topics_df.to_excel('/Users/annalisa/PycharmProjects/MSc_Project/topic_model/tm_con/con_topics_df9.xlsx')

# DOMINANT TOPIC FOR EACH DOCUMENT
corpus_topic_df = main_topic_doc_df(ldaMallet9, corpus, tweets_df)
corpus_topic_df.head(10)

# mapping the topic number to a topic name
topic_names = {1: 'Hunt Support', 2: 'Local Govt', 3: 'Custom Union', 4: 'Sport', 5: 'Foreign Policy',
               6: 'University/Innovation', 7: 'National Economy', 8: 'Infrastructure/Enterprise', 9: 'SNP'}
corpus_topic_df['topic_name'] = corpus_topic_df['main_topic'].map(topic_names)
corpus_topic_df.head(20)
corpus_topic_df.loc[corpus_topic_df['main_topic'] == 8]

# GROUP BY TOPICS
freq_topic_df = freq_topic_df(corpus_topic_df, corpus)
print(freq_topic_df)
pickle.dump(freq_topic_df, open('topic_model/tm_con/freq_topic_con.pkl', 'wb'), protocol=4)

freq_topic_df = pd.read_pickle('topic_model/tm_con/freq_topic_con.pkl')

# plot the frequency of topics in the corpus
freq_topic_df.sort_values(by=['total_docs'], ascending=False).plot.bar(x='topic_name', y='total_docs', legend=False)
plt.show()

# which document makes the highest contribution to each topic
corpus_topic_df.groupby('main_topic').apply(
    lambda topic_set: (topic_set.sort_values(by=['%_contribution'], ascending=False).iloc[0])).reset_index(drop=True)
