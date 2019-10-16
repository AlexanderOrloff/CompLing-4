# -*- coding: utf-8 -*-
from flask import Flask
from flask import render_template, request
from searchers import * #search_tf_idf, search_bm, search_elmo, search_fasttext
import logging


app = Flask(__name__)
logging.basicConfig(filename = 'my_app.log', level = logging.INFO)
logger = logging.getLogger()


@app.route('/')
@app.route('/index')
def index():
    return render_template('Index.html')

@app.route('/bm')
def bm_site():
    return render_template('bm.html')

@app.route('/bmresult')
def bm_result_site():
    if request.args:
        word = request.args['word']
        logger.info('for BM25 we GOT:')
        logger.info(word)
        res = search_bm(word, voc1, tf1)
        #logger.info('through BM25 we GAVE:', res) #!
        logger.info('through BM25 we GAVE:')
        logger.info(res)
    return render_template('bmresult.html', text = res)

@app.route('/tfidf')
def tfidf_site():
        return render_template('tfidf.html')

@app.route('/tfidfresult')
def tfidf_result_site():
    if request.args:
        word = request.args['word']
        logger.info('for TF_IDF we GOT:')
        logger.info(word)
        res = search_tf_idf(word, vocidf, tf_idf)
        logger.info('through TF_IDF we GAVE:')
        logger.info(res)
    return render_template('tfidfresult.html', text = res)

@ app.route('/elmo')
def elmo_site():
        return render_template('elmo.html')

@app.route('/elmoresult')
def elmo_result_site():
    if request.args:
        word = request.args['word']
        logger.info('for ELMO we GOT:')
        logger.info(word)
        res = search_elmo(word, elmo_vectors, sess, batcher, sentence_character_ids, elmo_sentence_input)
        logger.info('through ELMO we GAVE:')
        logger.info(res)
    return render_template('elmoresult.html', text=res)

@ app.route('/fasttext')
def fasttext_site():
        return render_template('fasttext.html')

@app.route('/fasttextresult')
def fasttext_result_site():
    if request.args:
        word = request.args['word']
        logger.info('for FASTTEXT we GOT:')
        logger.info(word)
        res = search_fasttext(word, vecs_fast, model_fast)
        logger.info('through FASTTEXT we GAVE:')
        logger.info(res)
    return render_template('fasttextresult.html', text = res)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded = False)
