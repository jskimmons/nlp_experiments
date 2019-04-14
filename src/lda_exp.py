import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import nltk
import ssl

# used to download nltk stuff
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('wordnet')

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

# dictionary.filter_extremes(no_below=5, no_above=0.6, keep_n= 100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_msgs]

lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 5, 
                                   id2word = dictionary,                                    
                                   passes = 15,
                                   workers = 2)

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

# topic 0: social media
# topic 1:
# topic 2:
# topic 3:
# topic 4: 

unseen_document = "We should put that in the pitch deck"

print(preprocess(unseen_document))

# bow_vector = dictionary.doc2bow(preprocess(unseen_document))

# for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
#     print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))