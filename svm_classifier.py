import pickle
import collections
import numpy as np
import pandas as pd
import spacy
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
from wordcloud import WordCloud



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


stop_words = ['xe2', 'xe', 'fetus', 'sv', 'ac', 'sydney', 'x80', 'user', 'abortion', 'xxxviii', 'kagan', 'parfit', 'oxford', 'new york university', 'midwest', '``', '[', '\'\'', '\\\\xe2', '&', 'user\\\\non', '0812', '2018', ']', '\\\\xe2\\\\x80\\\\x94', 'york', r'user\\\\non', 'user\\non', r'user\\non', r'\\xe2\\x80\\x94', r'\\\\xe2\\\\x80\\\\x94', 'id',
    '\\xe2\\x80\\x94', 'by new university', 'new university march', 'university march', '( ).',
    'donaldson', '<', ': )', 'jones', 'nora', 'university', 'march'  ]
for i in range(0, 3000):
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


def main():
    clf, vectorizer = getClassifierAndVectorizer()
    coef = clf.coef_[0].tolist()
    print(len(coef))
    top = 50
    predictors = []
    deon_dic = dict()
    cons_dic = dict()
    n_grams = training_n_grams
    print(len(n_grams))
    print(len(coef))
    for i in range(top):
        val = min(coef)
        index = coef.index(val)
        predictors.append([n_grams[index], val])
        cons_dic[n_grams[index]] = abs(val)
        n_grams.pop(index)
        coef.pop(index)
    for i in range(top):
        val = max(coef)
        index = coef.index(val)
        predictors.append([n_grams[index], val])
        deon_dic[n_grams[index]] = abs(val)
        n_grams.pop(index)
        coef.pop(index)

    for i in predictors:
        print (i ,"\n")

    print(cons_dic)
    print(deon_dic)
    wordcloud = WordCloud(background_color="white", height=400, width=800).fit_words(cons_dic)
    plt.figure(figsize=(6, 3), dpi=1000)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('cons_cloud.png', dpi=1000)
    '''
    wordcloud = WordCloud(background_color="white", height=400, width=800).fit_words(deon_dic)
    plt.figure(figsize=(6, 3), dpi=1000)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('deon_cloud.png', dpi=1000)
    '''




if __name__ == '__main__':
    main()

