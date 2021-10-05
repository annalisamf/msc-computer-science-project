import numpy as np
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import Phrases, CoherenceModel, LdaModel
from gensim.models.phrases import Phraser
from gensim.models.wrappers import LdaMallet

from matplotlib import pyplot as plt

mallet_path = r'/Users/annalisa/PycharmProjects/TopicModeling/newsgroup/mallet-2.0.8/bin/mallet'

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
spacy.load('en_vectors_web_lg', vocab=nlp.vocab)
nlp.max_length = 1500000

stopwords = nlp.Defaults.stop_words | {'re', 'rt', 'co', 'amp'}

words_to_exclude = ['glasgow', 'newcastle', 'durham', 'wrexham', 'swansea', 'bristol', 'hull', 'walthamstow', 'york',
                     'plymouth',
                     'battersea', 'cambridge', 'middlesbrough', 'wolverhampton', 'leeds', 'sunderland', 'portsmouth',
                     'croydon',
                     'cardiff', 'peterborough', 'oxford', 'edinburgh', 'greater_manchester', 'newport', "cardiff",
                     "salford",
                     'brighton', 'nottingham', 'slough', 'hartlepool', 'lancashire', 'ealing', 'stoke_trent', 'oldham',
                     'blackpool', 'north_east', 'west_midlands', 'lincoln', 'bardford', 'barnsley', 'leichester',
                     'hull_north',
                     'bolton', 'harrow', 'newport_west', 'stoke', 'welsh', 'scottish', 'sheffield', 'chorley',
                     'grimsby',
                    'ipswich', 'colchester', 'northampton', 'swindon', 'worchester', 'southampton', 'northumberland',
                    'kent',
                    'essex', 'plymouth', 'west_midlands', 'dover', 'watford', 'cornwall', 'wimbledon', 'plymouth',
                    'gloucester',
                    'norwich', 'telford', 'redditch', 'norfolk', 'ashford', 'crawley', 'southend', 'worcester',
                    'stevenage',
                    'shrewsbury', 'fulham', 'shropshire', 'chelsea', 'suffolk', 'salisbury', 'dorset', 'chelmsford',
                    'devon',
                    'worcestershire', 'south_west', 'yorkshire', 'derbyshire', 'bournemouth', 'manchester', 'morley',
                    'medway',
                    'lewisham', 'preston', 'stockton', 'newham', 'chesterfield',
                    'bradford', 'huddersfield', 'lancaster', 'morecambe', 'humpshire', 'sussex', 'stafford',
                    'liverpool', 'ogmore',
                    'cheltenham', 'birmingham', 'surrey', 'bath', 'sutton', 'brecon', 'hampshire', 'dewsbury',
                    'leicester', 'rotherham', 'stroud', 'gower', 'wigan', 'wakefield', 'bridgend', 'llanelli', 'bury', 'canterbury']


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


# takes a list of lemmatized words
def make_bigrams(text):
    bigram_model = Phraser(Phrases(text, min_count=5, threshold=100))
    return [bigram_model[doc] for doc in text]


# creates a list of coherence scores (umass and cv) for mallet models

def coh_mallet(max_topic, corpus, id2word, docs_bigrams):
    umass_mallet = []
    cv_mallet = []
    for nb_topics in range(1, max_topic + 1):
        lda = LdaMallet(mallet_path, corpus=corpus, id2word=id2word, num_topics=nb_topics, random_seed=1)
        cohm = CoherenceModel(model=lda, corpus=corpus, dictionary=id2word, coherence='u_mass').get_coherence()
        coh_cv = CoherenceModel(model=lda, texts=docs_bigrams, dictionary=id2word, coherence='c_v').get_coherence()
        umass_mallet.append(cohm)
        cv_mallet.append(coh_cv)
        print(nb_topics, " u_mass : ", cohm, " - c_v : ", coh_cv)
    return umass_mallet, cv_mallet


# creates a plot with coherence values
def plot_coh_scores(x, y, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Number of Topics")
    plt.ylabel(ylabel)
    plt.show()


# convert the mallet model to a regular LDA gensim model, in order to perform the visualization
def convertldaMalletToldaGen(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha)
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim


# creates a dataframe with the word-topics
def term_doc_df(lda, nr_words=20):
    # top 20 significant terms and their probabilities for each topic :
    topics = [[(term, round(wt, 3)) for term, wt in lda.show_topic(n, topn=nr_words)] for n in
              range(0, lda.num_topics)]

    # dataframe for term-topic matrix:
    topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics],
                             columns=['Term ' + str(i) for i in range(1, 21)],
                             index=['Topic ' + str(t) for t in range(1, lda.num_topics + 1)]).T
    return topics_df


# DOMINANT TOPIC FOR EACH DOCUMENT
def main_topic_doc_df(LdaMallet, corpus, df):
    # distribution of topics per each document
    tm_distribution = LdaMallet[corpus]
    # Dominant topic per each document
    sorted_topic_distr = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_distribution]

    # create an empty dataframe
    topics_docs_df = pd.DataFrame()
    # get the screen_names from the original dataframe
    topics_docs_df['screen_name'] = df.screen_name
    topics_docs_df['main_topic'] = [item[0] + 1 for item in sorted_topic_distr]
    topics_docs_df['%_contribution'] = [round(item[1] * 100, 2) for item in sorted_topic_distr]
    # corpus_topic_df['Topic_term'] = [topics_df.iloc[t[0]]['term_per_topic'] for t in corpus_topics]
    return topics_docs_df


# from a list of lemmatized tokens, returns dictionary and corpus to use with lda model
def dict_corpus(lemmas, excluded_words):
    # exclude some locations from the words/bigrams
    lemmas = [[word for word in sent if word not in excluded_words] for sent in lemmas]

    # create dictionary
    id2word = corpora.Dictionary(lemmas)

    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    # no_below : Keep tokens which are contained in at least `no_below` documents. no_above : Keep tokens which are
    # contained in no more than `no_above` documents (fraction of total corpus size, not an absolute number). keep_n :
    # Keep only the first `keep_n` most frequent tokens.
    id2word.filter_extremes(no_below=10, no_above=0.5)
    corpus = [id2word.doc2bow(doc) for doc in lemmas]
    return id2word, corpus


# GROUP BY TOPICS - from a dataframe with distribution of topics, returns the sum of main topic occurrence per each
# document
def freq_topic_df(corpus_topic_df, corpus):
    freq_topic_df = corpus_topic_df.groupby('topic_name').agg(
        total_docs=('topic_name', np.size),
        docs_perc=('topic_name', np.size)).reset_index()

    freq_topic_df['docs_perc'] = freq_topic_df['docs_perc'].apply(
        lambda row: round((row * 100) / len(corpus), 2))
    return freq_topic_df
