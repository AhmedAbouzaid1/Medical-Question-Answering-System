from nltk import tokenize
from operator import itemgetter
import math
from nltk.corpus import stopwords
import mysql.connector
import string
import pandas as pd
import tensorflow as tf
import numpy as np
import transformers
import sys
import collections

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

##Global Var
mycursor = mydb.cursor()
stop_words = set(stopwords.words('english'))
np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv("icliniqQAs.csv")
model = tf.keras.models.load_model('bert_(bilstm-bigru)_sim_model.h5')
model.summary()
batch_size = 1
max_length = 128  # Maximum length of input sentence to the model.
labels = ["contradiction", "entailment", "neutral"]
#######################################################################

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

class KG:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def KGanswer(self, sent):
        answer = False
        res_classify = self.classifier.classify(sent)
        print(res_classify)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        print(res_sql)
        final_answers = self.searcher.search_main(res_sql)
        print(final_answers)

        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)

def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    proba = model.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba


def tagsanswer(question):
    question = question.lower()
    question = question.translate(str.maketrans('', '', string.punctuation))

    total_words = question.split()
    total_word_length = len(total_words)
    # print(total_word_length)

    total_sentences = tokenize.sent_tokenize(question)
    total_sent_len = len(total_sentences)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1
    tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())

    # print(tf_score)

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

    query = "SELECT id FROM questiontags.tags WHERE tag = "
    for x in tags:
        query = query + "'" + x + "'" + " or tag = "

    query = query[:len(query) - 10]
    # print(query)
    mycursor.execute(query)

    myresult = mycursor.fetchall()

    Output = collections.defaultdict(int)

    for elem in myresult:
        Output[elem[0]] += 1

    # Printing output
    a = sorted(Output.items(), key=lambda x: x[1], reverse=True)[:3]
    print(a)
    res = []
    for x in a:
        q2 = df['question']
        q2 = q2[x[0] - 1]
        xx = check_similarity(question, q2)
        if (xx[0] != 'contradiction'):
            res.append(xx[1])

    index = res.index(max(res))
    index = a[index]
    # print(index[0])
    Ans = df["answer"]
    answer = Ans[index[0] - 1]

    return answer, index[0] - 1


def accuracy():
    counter_max = 0
    #kng = KG()
    for num, question in enumerate(df['question'][:10]):
        #answer = kng.KGanswer(question)

        # pred, prob = check_similarity(question, answer)
        # print(pred, " ", prob)
        # if (pred == "contradiction"):
        #     answer, index = tagsanswer(question)
        print(question)
        #if answer == False:
        answer, index = tagsanswer(question)
        print (index)
        if index == num:
            counter_max += 1
        #else:
            #print(question, answer)
            #counter_max += 1
    return counter_max


def main():
    print(accuracy())

if __name__ == '__main__':
    main()
