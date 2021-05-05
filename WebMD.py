import json
from nltk import tokenize
from operator import itemgetter
import math
import nltk
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

with open('C://Users\CSE-P07-2179-G9\Documents\GitHub\Medical-Question-Answering-System\webmdQAs.json') as f:
  data = json.load(f)

id = 466
for i in data:
    for tag in i['tags']:
        sql = "INSERT INTO tags (id, tag) VALUES (%s, %s)"
        val = (id, tag)
        mycursor.execute(sql, val)
        mydb.commit()
    id = id + 1

