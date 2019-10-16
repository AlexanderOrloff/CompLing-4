import csv
import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer()


def get_texts():
    corpus = []
    inquiery = []
    lenghths = []
    scores = []
    normal = []

    with open('quora_question_pairs_rus.csv', encoding = 'utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            normal.append(row[2])
            text = preproc(row[2])
            corpus.append(text)
            inquiery.append(row[1])
            lenghths.append(len(text.split(' ')))
            scores.append(row[3])
            print(len(scores))
            if len(scores) == 25000:
                return [corpus, inquiery, lenghths, scores, normal]


def preproc(data):
            data = data.split()
            text = ''
            for word in data:
                word = word.strip('[!,.?"]')
                p = morph.parse(word.strip())[0]
                p = p.normal_form
                if p != '-':
                    if re.search(r'\d', p) == None:
                        text = text + ' ' + p
            return text[1:]

data = get_texts()
corpus_texts, inquiery_texts, lenghths, scores, normal_texts =  data[0][1:], data[1][1:], data[2][1:], data[3][1:], data[4][1:]
del data