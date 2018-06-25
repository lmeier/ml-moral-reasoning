from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pickle
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from time import time


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

#__________________________________




def loadDataset(pklPath):
    with open(pklPath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))

def splitData(startString, dataSet, divisions=100):
    newString = startString
    for i in dataSet:
        newString= newString + str(i)
    #return newString
    newList = []
    div = len(newString)//100
    temp = ""
    for j in range(len(newString)):
        temp = temp + newString[j]
        if j%div == 0:
            newList.append(temp)
            temp = ""
    newList.append(temp)
    return newList

c_data = splitData("", loadDataset("consPapersNew.pkl"))
d_data = splitData("", loadDataset("deonPapersNew.pkl"))
final_data_set = c_data  + d_data

y = []
for i in c_data:
    y.append('cons')
for i in d_data:
    y.append('deon')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


stop_words = ['xe2', 'xe', 'fetus', 'sv', 'ac', 'sydney', 'x80', 'user', 'abortion', 'xxxviii', 'kagan', 'parfit', 'oxford', 'new york university', 'midwest', '``', '[', '\'\'', '\\\\xe2', '&', 'user\\\\non', '0812', '2018', ']', '\\\\xe2\\\\x80\\\\x94', 'york', r'user\\\\non', 'user\\non', r'user\\non', r'\\xe2\\x80\\x94', r'\\\\xe2\\\\x80\\\\x94', 'id',
    '\\xe2\\x80\\x94', 'by new university', 'new university march', 'university march', '( ).',
    'donaldson', '<', ': )', 'jones', 'nora', 'university', 'march'  ]
for i in range(0, 3000):
    stop_words.append(str(i))

vectorizer = TfidfVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', tokenizer=LemmaTokenizer(), stop_words=stop_words, strip_accents='ascii', max_df=.7, )

X = vectorizer.fit_transform(final_data_set)

nComponents = 20

lda = LatentDirichletAllocation( max_iter=5,
                                learning_method='online',
                                learning_offset=50,
                                random_state=0)

lda.fit(X)

n_top_words = 10
print("\nTopics in LDA model:")
tf_feature_names = vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
