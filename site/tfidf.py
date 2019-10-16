from data_col import *

#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


vectorizeridf = TfidfVectorizer()
vectorizer1 = CountVectorizer(binary = True)
vectorizertf = TfidfVectorizer(use_idf=False)

tf_idf = vectorizeridf.fit_transform(corpus_texts)
vocidf = vectorizeridf.get_feature_names()
tf_idf = tf_idf.toarray()

avgdl = np.mean(lenghths)
k = 2
b = 0.75

corpus_matrix1 = vectorizer1.fit_transform(corpus_texts)
voc1 = vectorizer1.get_feature_names()
N = len(corpus_matrix1.getnnz(axis = 1))
n = corpus_matrix1.sum(axis=0)
n = n.tolist()[0]

idf = []
for i in range(len(voc1)):
    idf_p = log((N-n[i]+0.5) / (n[i]+0.5))
    if idf_p < 0:
            idf_p = 0
    idf.append(idf_p)

del corpus_matrix1

tf1 = vectorizertf.fit_transform(corpus_texts)
tf1 = tf1.toarray()

additional = []
for i in lenghths:
    res = k * (1-b+b*(i/avgdl))
    additional.append(res)

k1 = k + 1
for i in range(N):
    for j in range(len(voc1)):
        tf1[i][j] = (tf1[i][j]*k1)/(tf1[i][j] + additional[i]) * idf[j]

del additional
