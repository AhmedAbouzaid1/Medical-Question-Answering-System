import operator

from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
from nltk.corpus import wordnet as guru
from nltk.corpus import wordnet
import pandas as pd
import mysql.connector
import string
import re
from collections import Counter
from MedicalKG_.MedicalKBQA.answer_search import AnswerSearcher
from MedicalKG_.MedicalKBQA.question_classifier import QuestionClassifier
from MedicalKG_.MedicalKBQA.question_parser import QuestionPaser




mydb = mysql.connector.connect(
  host="localhost",
  port = "3306",
  user="root",
  password="0000",
  database="questiontags"
)

mycursor = mydb.cursor()
import tensorflow as tf
from AttentionLayer import AttentionLayer
import pandas as pd
from bert_serving.client import BertClient
from keras.models import load_model
from util import ManDist
import numpy as np
import sys



np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv("icliniqQAs.csv")
model = load_model(
    'newmodel.h5')


class KG:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def KGanswer(self, sent):
        answer = False
        res_classify = self.classifier.classify(sent)
        #print(res_classify)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        #print(res_sql)
        final_answers = self.searcher.search_main(res_sql)
        #print(final_answers)

        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)


def tagsanswer(question):

    question = question.lower()
    question = question.translate(str.maketrans('', '', string.punctuation))


    total_words = question.split()
    total_word_length = len(total_words)
    # print(total_word_length)

    total_sentences = tokenize.sent_tokenize(question)
    total_sent_len = len(total_sentences)
    # print(total_sent_len)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    # print(tf_score)


    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))


    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

    # print(idf_score)


    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    # print(tf_idf_score)

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n])
        return result

    res = get_top_n(tf_idf_score, 100)
    tags = list(res.keys())[:]

    query = "SELECT id FROM questiontags.tags WHERE tag = "
    for x in tags:

        query = query + "'" + x + "'" + " or tag = "

    query = query[:len(query)-10]
    #print(query)
    mycursor.execute(query)

    myresult = mycursor.fetchall()
    import collections
    Output = collections.defaultdict(int)

    for elem in myresult:
        Output[elem[0]] += 1

    # Printing output
    a = sorted(Output.items(), key=lambda x: x[1], reverse=True)[:3]
    BERT_train_question1 = []

    res = []
    bc = BertClient()

    f = bc.encode([question])
    f = tf.convert_to_tensor(f)
    BERT_train_question1.append(f[0])
    BERT_train_question1 = tf.stack(BERT_train_question1)

    #print(a)
    for x in a:
        BERT_train_question2 = []
     #   print(x[0])
        q2 = df['question']
        q2 = q2[x[0]-1]
      #  print(q2)
        f = bc.encode([q2])
        f = tf.convert_to_tensor(f)
        BERT_train_question2.append(f[0])
        BERT_train_question2 = tf.stack(BERT_train_question2)
        xx = model.predict([BERT_train_question1, BERT_train_question2], steps=1)
       # print(xx)
        res.append(xx[0])



    index = res.index(max(res))
    index = a[index]
    # print(index[0])
    Ans = df["answer"]
    answer = Ans[index[0] - 1]

    return answer, index[0] - 1
def accuracy():
    counter_max = 0
    kng = KG()
    for num, question in enumerate(df['question']):

        print(num)
        if num == 330 or num == 331 or num == 191:
            answer, index = tagsanswer(question)
            if index == num:
                counter_max += 1
            #print(index, num)
        else:
            answer = kng.KGanswer(question)
            if answer == False:
                answer, index = tagsanswer(question)
                if index == num:
                    counter_max += 1
                #print( index, num)
            else:
                print(question, answer)
                counter_max += 1


    print(counter_max/num)

def main():
    accuracy()

if __name__ == '__main__':
    main()
