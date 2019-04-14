import os

from collections import defaultdict

import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import common_texts, get_tmpfile
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec, Phrases


from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import nltk
from nltk.cluster import KMeansClusterer

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

# iterate through all given messages and preprocess them
processed_msgs = []

for l in open('general_text.txt', 'r'):
    l = l.strip()
    p = preprocess(l)
    if p:
        processed_msgs.append(p)

dictionary = gensim.corpora.Dictionary(processed_msgs)

# print(dictionary)

exists = os.path.isfile("/word2vec.model")
if not exists:
    path = get_tmpfile("word2vec.model")
    bigram_transformer = Phrases(processed_msgs)
    model = Word2Vec(bigram_transformer[processed_msgs], min_count=1)
    model.save("word2vec.model")
else:
    model = Word2Vec.load("word2vec.model")


# w2v_corpus = model.wv
# print(w2v_corpus.vocab)
# del model


X = model[model.wv.vocab]

NUM_CLUSTERS=10
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
# print(assigned_clusters)

# words = list(model.wv.vocab)
# for i, word in enumerate(words):  
#     print (word + ":" + str(assigned_clusters[i]))

words = list(model.wv.vocab)
cluster_dict = defaultdict(lambda: [])
for i, word in enumerate(words):
    cluster_dict[assigned_clusters[i]].append(word)


print(cluster_dict[4])