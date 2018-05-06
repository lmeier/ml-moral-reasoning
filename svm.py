import pickle
import collections
import numpy as np
import pandas as pd
import spacy
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


""""
Load training data
"""

#http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

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


stop_words = ['xe2', 'xe', 'fetus', 'sv', 'ac', 'sydney', 'x80', 'user', 'abortion', 'xxxviii', 'kagan', 'parfit', 'oxford', 'new york university', 'midwest', '``', '[', '\'\'', '\\\\xe2', '&', 'user\\\\non', '0812', '2018', ']', '\\\\xe2\\\\x80\\\\x94', 'york', r'user\\\\non', 'user\\non', r'user\\non', r'\\xe2\\x80\\x94', r'\\\\xe2\\\\x80\\\\x94']
for i in range(0, 100):
    stop_words.append(str(i))

vectorizer = TfidfVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', tokenizer=LemmaTokenizer(), stop_words=stop_words, strip_accents='ascii', max_df=.7, )

X = vectorizer.fit_transform(final_data_set).toarray()
training_n_grams = vectorizer.get_feature_names()
#tfidf = TfidfVectorizer(stop_words ='english' , max_df=.5, ngram_range=(1,5))
#X = tfidf.fit_transform(final_data_set).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 99)

print("SVM with Vector Featues")
clf = sklearn.svm.LinearSVC().fit(X_train, y_train)

def getClassifierAndVectorizer():
    """
    Make sure to pass all data to be predicted
    """
    print("imported correctly")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', tokenizer=LemmaTokenizer(), stop_words=stop_words, strip_accents='ascii', max_df=.7, )
    X = vectorizer.fit_transform(final_data_set)
    training_n_grams = vectorizer.get_feature_names()
    clf = sklearn.svm.LinearSVC().fit(X, y)
    return clf, vectorizer

"""
print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)


    return clf, n_grams, fullX, position_of_new_X
"""

