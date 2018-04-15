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

## const
PKL_PATH = "consPapersNew.pkl"



LEMMA_FILTER = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'] # keep 'NUM' as the last, function lemma_dep_list() refers to this
def loadDataset(pklPath=PKL_PATH):
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

c_data = splitData("", loadDataset("consPapersNew.pkl")) + splitData("", loadDataset("JSTORconsPapers.pkl"))
d_data = splitData("", loadDataset("deonPapersNew.pkl")) + splitData("", loadDataset("JSTORdeonPapers.pkl"))
final_data_set = c_data  + d_data


#implementing n-gram
bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
#print(analyze(c_data_set[0]))
#print(analyze(final_data_set))


X = bigram_vectorizer.fit_transform(final_data_set).toarray()
n_grams = bigram_vectorizer.get_feature_names()
tfidf = TfidfVectorizer(ngram_range=(1,3))
X = tfidf.fit_transform(final_data_set).toarray()

y = []
for i in c_data:
    y.append('cons')
for i in d_data:
    y.append('deon')


#Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 99)
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

#visualizing coefficients
coef = clf.coef_[0].tolist()
print(len(coef))
top = 100
predictors = []
for i in range(top):
    val = min(coef)
    index = coef.index(val)
    predictors.append([n_grams[index], val])
    coef.pop(index)
for i in predictors:
    print (i ,"\n")

#check accuracy of NB

n = 0
correct = 0
for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))
    n = n + 1
    if i == j:
        correct = correct + 1

print(correct*100/n)

#SVM
print("SVM with Vector Featues")
clf = sklearn.svm.LinearSVC().fit(X_train, y_train)
predicted = clf.predict(X_test)
n = 0
correct = 0
for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))
    n = n + 1
    if i == j:
        correct = correct + 1
print(correct*100/n)
'''
