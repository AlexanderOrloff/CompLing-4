from tfidf import *
from gensim.models import Word2Vec, KeyedVectors

model_file = './fast/model.model'
model_fast = KeyedVectors.load(model_file)

vecs = []
for text in corpus_texts:
    lenghth = 0
    res = np.zeros((model_fast.vector_size,), dtype = 'float32')
    for word in text.split(' '):
        if word in model_fast.wv.vocab:
            lenghth += 1
            res += model_fast.wv[word]
    res = res/lenghth
    vecs.append(res)

vecs_fast = np.nan_to_num(vecs)
