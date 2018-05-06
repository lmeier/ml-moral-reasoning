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


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



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

stop_words = ['xe2', 'xe', 'fetus', 'sv', 'ac', 'sydney', 'x80', 'user', 'abortion', 'xxxviii', 'kagan', 'parfit', 'oxford', 'new york university', 'midwest', '``', '[', '\'\'', '\\\\xe2', '&', 'user\\\\non', '0812', '2018', ']', '\\\\xe2\\\\x80\\\\x94', 'york', r'user\\\\non', 'user\\non', r'user\\non', r'\\xe2\\x80\\x94', r'\\\\xe2\\\\x80\\\\x94']
for i in range(0, 100):
    stop_words.append(str(i))
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', tokenizer=LemmaTokenizer(), stop_words=stop_words, strip_accents='ascii', max_df=.7, )
#analyze = bigram_vectorizer.build_analyzer()

X = bigram_vectorizer.fit_transform(final_data_set) #.toarray()
n_grams = bigram_vectorizer.get_feature_names()
#tfidf = TfidfVectorizer(stop_words ='english' , max_df=.5, ngram_range=(1,5))
#X = tfidf.fit_transform(final_data_set).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 99)

print("SVM with Vector Featues")
clf = sklearn.svm.LinearSVC().fit(X_train, y_train)
predicted = clf.predict(X_test)
probs = clf.decision_function(X_test)
print(probs)
n = 0
correct = 0
for i, j, k in zip(y_test, predicted, probs):
    print('%r => %s, %s' % (i, j, k))
    n = n + 1
    if i == j:
        correct = correct + 1
print(correct*100/n)

coef = clf.coef_[0].tolist()
print(len(coef))
top = 100
predictors = []
print(len(n_grams))
print(len(coef))
for i in range(top):
    val = min(coef)
    index = coef.index(val)
    predictors.append([n_grams[index], val])
    n_grams.pop(index)
    coef.pop(index)
for i in range(top):
    try:
        val = max(coef)
        index = coef.index(val)
        predictors.append([n_grams[index], val])
        n_grams.pop(index)
        coef.pop(index)
    except:
        #print("error")
        x = 1
for i in predictors:
    print (i ,"\n")
