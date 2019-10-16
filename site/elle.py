import csv
import os
from math import log
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from string import punctuation
punctuation += '…—'
from pymorphy2 import MorphAnalyzer
import time
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import sys
from sklearn.metrics.pairwise import cosine_similarity
#import warnings
#warnings.filterwarnings('ignore')
# ~ global variables
pymorphy2_analyzer = MorphAnalyzer()
rus_stopwords = stopwords.words('russian')


with open('quora_question_pairs_rus.csv', 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    # skip the header
    print('header:', next(csv_reader, None))
    queries = []
    docs = []
    answers = []
    for row in csv_reader:
        for _, query, doc, answer in csv_reader:
            queries.append(query)
            docs.append(doc)
            answers.append(int(float(answer)))

def preprocess_text(text, save_stopwords=True):
    lowered_tokens = [word.strip(punctuation) for word in word_tokenize(text.lower()) if word.strip(punctuation)]
    if save_stopwords:
        return [pymorphy2_analyzer.normal_forms(token)[0] for token in lowered_tokens]
    return [pymorphy2_analyzer.normal_forms(token)[0] for token in lowered_tokens if pymorphy2_analyzer.normal_forms(token)[0] not in rus_stopwords]

N = 2000
tokenized_docs = [preprocess_text(doc) for doc in docs[:N]]

from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_elmo_embeddings(directory, top=False):
    """
    :param directory: directory with an ELMo model ('model.hdf5', 'options.json' and 'vocab.txt.gz')
    :param top: use ony top ELMo layer
    :return: ELMo batcher, character id placeholders, op object
    """
    vocab_file = os.path.join(directory, 'vocab.txt.gz')
    options_file = os.path.join(directory, 'options.json')
    weight_file = os.path.join(directory, 'model.hdf5')

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, 50)

    # Input placeholders to the biLM.
    sentence_character_ids = tf.placeholder('int32', shape=(None, None, 50))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=300)

    # Get ops to compute the LM embeddings.
    sentence_embeddings_op = bilm(sentence_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=top)
    return batcher, sentence_character_ids, elmo_sentence_input

def get_elmo_vectors(sess, texts, batcher, sentence_character_ids, elmo_sentence_input):
    """
    :param sess: TensorFlow session
    :param texts: list of sentences (lists of words)
    :param batcher: ELMo batcher object
    :param sentence_character_ids: ELMo character id placeholders
    :param elmo_sentence_input: ELMo op object
    :return: embedding matrix for all sentences (max word count by vector size)
    """

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print('Sentences in this batch:', len(texts), file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})

    return elmo_sentence_input_

batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(os.path.join('elmo'))

elmo_vector_size = 1024
elmo_vectors = np.zeros((len(tokenized_docs), elmo_vector_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 300
    for i in range(0, len(tokenized_docs), batch_size):
        elmo_vectors[i:i + batch_size] = np.mean(get_elmo_vectors(
            sess, tokenized_docs[i:i + batch_size], batcher, sentence_character_ids, elmo_sentence_input), axis=1)



#results = search_elmo('почему я чувствую, что я взволнован чем-то', elmo_vectors, sess, batcher, sentence_character_ids, elmo_sentence_input)
#print('\n'.join([f'{i + 1} - {elem}' for i, elem in enumerate(results[:5])]))