import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from vectoring import *
from elle import *
from scipy import spatial


def search_bm(inquiery, voc, tf1):
    inq = preproc(inquiery)
    vec = np.zeros((len(voc)))
    for word in inq.split(' '):
        if word in voc:
            index = voc.index(word)
            vec[index] = 1
    res = np.dot(tf1, vec)
    results = np.argsort(res)[::-1][:10]
    r = '\n'.join([f"{str(digit + 1)}. {normal_texts[i].capitalize()}" for digit, i in enumerate(results)])
    return r

def search_tf_idf(inquiery, voc, tf_idf):
    inq = preproc(inquiery)
    vec = np.zeros((len(voc)))
    for word in inq.split(' '):
        if word in voc:
            index = voc.index(word)
            vec[index] = 1
    res = np.dot(tf_idf,vec)
    results = np.argsort(res)[::-1][:10]
    r = '\n'.join([f"{str(digit + 1)}. {normal_texts[i].capitalize()}" for digit, i in enumerate(results)])
    return r


def inquiery_vector_fast(inq, model_fast):
    inq_vec = np.zeros((model_fast.vector_size,), dtype = 'float32')
    inq_l = 0
    for word in preproc(inq).split():
        if word in model_fast.wv.vocab:
            inq_l += 1
            inq_vec += model_fast.wv[word]
    inq_vec = inq_vec/inq_l
    return inq_vec


def search_fasttext(inq, vecs_fast, model_fast):
    inq_vec = inquiery_vector_fast(inq, model_fast).reshape(-1, 1)
    results = {}
    for index, text in enumerate(vecs_fast):
        a = text.reshape(-1, 1)
        res = 1 - spatial.distance.cosine(a, inq_vec)
        results[index] = res
    results_sorted = sorted(results, key=results.get, reverse=True)[:10]
    r = '\n'.join([f"{str(digit + 1)}. {normal_texts[i].capitalize()}" for digit, i in enumerate(results_sorted)])
    return r


def get_most_probable_docs(doc_vector, docs_matrix):
    cosine_values = cosine_similarity(docs_matrix, doc_vector.reshape(1, -1)).reshape(docs_matrix.shape[0])
    return [docs[doc_id] for doc_id, _ in sorted(list(
        enumerate(cosine_values)), key=lambda elem: elem[1], reverse=True)]


def search_elmo(query, elmo_vectors, sess, batcher, sentence_character_ids, elmo_sentence_input):
    tokenized_query = preprocess_text(query)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        elmo_vector = np.mean(get_elmo_vectors(
            sess, [tokenized_query], batcher, sentence_character_ids, elmo_sentence_input), axis=1)[0]
    results = get_most_probable_docs(elmo_vector, elmo_vectors)
    res = '\n'.join([f'{i + 1}. {elem.capitalize()}' for i, elem in enumerate(results[:10])])
    return res
