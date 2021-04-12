from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
from nltk.corpus import wordnet as guru
from nltk.corpus import wordnet
import pandas as pd
import mysql.connector
import string

mydb = mysql.connector.connect(
    host="localhost",
    port="3306",
    user="root",
    password="0000",
    database="questiontags"
)

mycursor = mydb.cursor()

df = pd.read_csv(
    "icliniqQAs.csv")
Q1 = df["question"]
id = 1
for i in Q1:
    i = i.lower()
    i = i.translate(str.maketrans('', '', string.punctuation))

    total_words = i.split()
    total_word_length = len(total_words)
    # print(total_word_length)

    total_sentences = tokenize.sent_tokenize(i)
    total_sent_len = len(total_sentences)
    # print(total_sent_len)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())
    print(tf_score)


    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))


    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())

    # print(idf_score)

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}


    # print(tf_idf_score)

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result


    res = get_top_n(tf_idf_score, 100)
    tags = list(res.keys())[:]
    print("The question tags: ")

    ##################################################################################

    # Extracting synonyms

    # for tag in tags:

    #     syns = wordnet.synsets(tag)
    #     import re
    #     syns = str(syns)
    #     res = re.findall(r'\(.*?\)', syns)
    #     # printing result
    #     for i in res:
    #         x = i.split("'")[1].split(".")[0]
    #         if not x in tags:
    #             tags.append(str(x))
    #         # print(syn)
    from itertools import chain

    x = set(())
    for tag in tags:
        synonyms = wordnet.synsets(tag)
        lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        x = x | lemmas

    tags += x
    for x in tags:
        sql = "INSERT INTO tags (id, tag) VALUES (%s, %s)"
        val = (id, x)
        mycursor.execute(sql, val)
        mydb.commit()
    id = id + 1
