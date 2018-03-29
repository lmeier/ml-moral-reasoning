import numpy as np
import glob, os, pickle
import math

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm





def loadDataset(filePath):
    with open(filePath, "rb") as pklFile:
        return np.array(pickle.load(pklFile, encoding="utf-8"))


f1 = '/Users/liammeier/moral-reasoning/consPapers.pkl'
f2 = '/Users/liammeier/moral-reasoning/deonPapers.pkl'

consArray = loadDataset(f1)
deonArray = loadDataset(f2)

X = []
y = []
for i in consArray:
    X.append(i)
    y.append('cons')
for i in deonArray:
    X.append(i)
    y.append('deon')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 50)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)

for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))

#Now running Bayes with new features (see featureExtraction.py)
print("Naive Bayes with Vector Features")
def loadDataset(filePath):
    with open(filePath, "rb") as pklFile:
        return pickle.load(pklFile, encoding="utf-8")

X = loadDataset('/Users/liammeier/moral-reasoning/vecX.pkl')
y = loadDataset('/Users/liammeier/moral-reasoning/classY.pkl')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 50)

clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))


print("SVM with Vector Featues")

clf = sklearn.svm.LinearSVC().fit(X_train, y_train)
predicted = clf.predict(X_test)

for i, j in zip(y_test, predicted):
    print('%r => %s' % (i, j))


