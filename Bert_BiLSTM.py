# -*- coding: utf-8 -*-
# @Author  : Bill Bao
# @File    : train.py
# @Software: PyCharm and Spyder
# @Environment : Python 3.6+
# @Reference1 : https://zhuanlan.zhihu.com/p/31638132
# @Reference2 : https://github.com/likejazz/Siamese-LSTM
# @Reference3 : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM

# 基础包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys



import tensorflow as tf
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import numpy as np
from gensim.models import KeyedVectors
from keras import initializers as initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, Bidirectional, GlobalAveragePooling1D,TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D,GlobalMaxPooling1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K
from keras.engine import Layer
# from tf.compat.v1.keras import backend as K
# K.set_session()

# import tensorflow.compat.v1 as tf
import keras.layers as layers

# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# from BertTuned import get_features
# tf.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()
# from AttentionLayer import AttentionLayer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from AttentionLayer import AttentionLayer
from util import make_w2v_embeddings, split_and_zero_padding, ManDist
from bert_serving.client import BertClient





'''pip
本配置文件用于训练孪生网络
'''

# ------------------预加载------------------ #






# TRAIN_CSV='C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject/quora_duplicate_questions.tsv'

flag = 'en'
# embedding_path = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/GoogleNews-vectors-negative300 .bin.gz'
# max_seq_length = 10
max_seq_length = 25
savepath = 'newxxmodel.h5'
# # train_dff = pd.read_csv(TRAIN_CSV, sep="\t")
# # train_dff.to_csv('C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject/uncased_L-12_H-768_A-12/train.csv',index=False)
#
# TRAIN_CSVV='C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject/uncased_L-12_H-768_A-12/train.csv'
# train_df = pd.read_csv(TRAIN_CSVV,  encoding = "ISO-8859-1")
# print('loaded data')




train_df = pd.read_csv("C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject1/SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
valid_df = pd.read_csv("C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject1/SNLI_Corpus/snli_1.0_dev.csv")
test_df = pd.read_csv("C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject1/SNLI_Corpus/snli_1.0_test.csv")

train_df = train_df[:20000]
valid_df =valid_df[:20000]
test_df =test_df[:20000]



# X = train_df[['question1_n', 'question2_n']]



def text_to_word_list(flag, text):  # 文本分词
    text = str(text)
    text = text.lower()

    if flag == 'cn':
        pass
    else:
        # 英文文本下的文本清理规则
        import re
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text



# train_df["label"] = train_df["similarity"].apply(
#     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
# )
# y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)
#
#
# X = train_df[['question1', 'question2']]
# X=X[0:90000]
#
# Y = train_df['is_duplicate']
# Y=Y[0:90000]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
# # print(X_train.head())
#
# X= X_train[['question1', 'question2']]# dataset shortened
# X_v= X_validation[['question1', 'question2']]# dataset shortened
#
# print("loaded training and validation data")
# Y_train=Y_train
# # print(Y_train)
# Y_validation=Y_validation0

# print(X)

labels = ["contradiction", "entailment", "neutral"]


# There are more than 550k samples in total; we will use 100k for this example.
# train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
# valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
# test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")
# train_df = train_df[:10000]
# valid_df = valid_df[:10000]
# test_df = test_df[:1000]
# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")

print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
print(f"Similarity: {train_df.loc[1, 'similarity']}")

# We have some NaN entries in our train data, we will simply drop them.
print("Number of missing values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)

print("Train Target Distribution")
print(train_df.similarity.value_counts())


print("Validation Target Distribution")
print(valid_df.similarity.value_counts())

train_df = (
    train_df[train_df.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)
valid_df = (
    valid_df[valid_df.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

train_df["label"] = train_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

valid_df["label"] = valid_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)

test_df["label"] = test_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)


#train_question1
bc = BertClient()
BERT_train_question1=[]
k=0



for x in train_df['sentence1']:



    # print("text is",x)
    # f= elmo([x], signature="default", as_dict=True)["elmo"]   #ater fixing it dontforget to change it to the main dataset
    # print ("before",f)
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_train_question1.append(f[0])
    k=k+1
    print("BERT_train_question1: point ",k)


BERT_train_question1 = tf.stack(BERT_train_question1)
print("Done BERT_train_question1",BERT_train_question1.shape)
# print("Done BERT_train_question1",np.shape(BERT_train_question1))

#train_question2
k=0
BERT_train_question2=[]
for x in train_df['sentence2']:
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_train_question2.append(f[0])



    k = k + 1
    print("BERT_train_question2: point ", k)

BERT_train_question2 = tf.stack(BERT_train_question2)
print("Done BERT_train_question2",BERT_train_question2.shape)


#test_question1
k=0
BERT_test_question1=[]
for x in valid_df['sentence1']:
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_test_question1.append(f[0])

    k = k + 1
    print("BERT_test_question1: point ", k)

BERT_test_question1 = tf.stack(BERT_test_question1)
print("Done BERT_test_question1",BERT_test_question1.shape)
# print("Done BERT_test_question1",np.shape(BERT_test_question1))

#test_question2
k=0
BERT_test_question2=[]
for x in valid_df['sentence2']:
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_test_question2.append(f[0])

    k = k + 1

    print("BERT_test_question2: point ", k)

BERT_test_question2 = tf.stack(BERT_test_question2)
print("Done BERT_test_question2",BERT_test_question2.shape)
# print("Done BERT_test_question2",np.shape(BERT_test_question2))








# X_train = split_and_zero_padding(X_train, max_seq_length)
# X_validation = split_and_zero_padding(X_validation, max_seq_length)

# 将标签转化为数值

# class ElmoEmbeddingLayer ( Layer ):
#     def __init__(self, **kwargs):
#         self.dimensions = 1024
#         self.trainable=True
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
#                                name="{}_module".format(self.name))
#
#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)
#
#     def call(self, x, mask=None):
#         result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                       as_dict=True,
#                       signature='default',
#                       )['default']
#         return result
#
#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '--PAD--')
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.dimensions)



# 确认数据准备完毕且正确


# -----------------基础函数------------------ #


# def shared_model_HBDA(_input):

    # 词向量化
    # embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
    #                     trainable=False)(_input)

    # embedding_layer = Embedding(len(embeddings) + 1,
    #                             embedding_dim,
    #                             input_length=max_seq_length)
    #
    # print(embedding_layer)
    # print(type(embedding_layer))
    #
    # embedded_sequences = embedding_layer(_input)
    # print('embedded sequence is ', embedded_sequences)
    # print('this is the first embedding layer', embedded_sequences)
    # print('this is size of the first embedding layer', embedded_sequences.shape())

    # input_layer = Input(shape=(1,), dtype="string", name="Input_layer")
    # print('here 1')
    # embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024,))(_input)
    # # embedding_layer =ElmoEmbeddingLayer()(_input)
    # print('here 2 after embedding_layer')
    # print(embedding_layer.shape)
    # BiLSTM = Bidirectional(layers.LSTM(1024, return_sequences=False, recurrent_dropout=0.2, dropout=0.2),name="BiLSTM")(embedding_layer)
    # # print('here 3 after bilstm')
    #
    # Dense_layer_1=Dense(8336, activation='relu')(BiLSTM)
    # Dropout_layer_1 = Dropout(0.5)(Dense_layer_1)
    # Dense_layer_2 = Dense(4168, activation='relu')(Dropout_layer_1)
    # Dropout_layer_2 = Dropout(0.5)(Dense_layer_2)
    # output_layer = Dense(1, activation='sigmoid')(Dropout_layer_2)


    # dense = Dense(256, activation='relu')(embedding_layer)
    # print(dense.shape)
    # BiLSTM = Bidirectional(LSTM(100, return_sequences=True))(dense)
    #
    # l_dense = TimeDistributed(Dense(200))(BiLSTM)
    # print("here: ", type(l_dense))
    #
    # # l_att = AttentionLayer()(l_dense)
    # print('here 4 after attention')

    # 单层Bi-LSTM
    # activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(embedded)

    # dropout
    # activations = Dropout(0.5)(activations)

    # Words level attention model
    # word_dense = Dense(1, activation='relu', name='word_dense')(activations)
    # word_att,word_coeffs = AttentionLayer(EMBED_SIZE,True,name='word_attention')(word_dense)

    # Attention
    # attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(n_hidden * 2)(attention)
    # attention = Permute([2, 1])(attention)
    # sent_representation = dot([activations, attention],axes=1)
    # dropout
    # sent_representation = Dropout(0.1)(sent_representation)

    # return output_layer


def shared_model_HBDA(_input):



    # print('this is size of the first embedding layer', embedded_sequences.shape())
    print(_input.shape)
    # out = Lambda(lambda x: x[0])(_input)
    # print(out.shape)

    print("Embedding Layer")


    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(_input)
    print("Bi-Directional LSTM Layer")
    l_dense = TimeDistributed(Dense(200))(l_lstm)

    l_att = AttentionLayer()(l_dense)
    print("Attetion Layer", l_att)
    # l_att=tf.reshape(l_att,shape=(-1,1,200))
    print("Attetion Layer2",l_att)


    return l_att


# -----------------主函数----------------- #

if __name__ == '__main__':
    # 超参
    batch_size = 2
    senrep=[]
    n_epoch = 100
    n_hidden = 50
    left_input = Input(shape=(max_seq_length,768,), dtype="float32",name="Input_layer")
    print("left", left_input.shape)
    # left_input = Lambda(lambda x: x[:, 0],)(left_input)
    # left_input = tf.reshape(left_input, shape=[-1, max_seq_length, 768])
    print("left2",left_input.shape)
    # left_input = Input(shape=(max_sequence_length,), dtype='float32')
    # print('this is left inout', left_input)
    right_input = Input(shape=(max_seq_length,768,), dtype="float32")
    # right_input = Lambda(lambda x: x[:, 0])(right_input)
    # right_input = tf.reshape(right_input, shape=[-1, max_seq_length, 768])
    print("right",right_input.shape)
    # print('this is right inout', right_input)
    # right_input = Input(shape=(1,), dtype='float32')
    left_sen_representation = shared_model_HBDA(left_input)
    print('left snetcen presentation', left_sen_representation)
    right_sen_representation = shared_model_HBDA(right_input)
    sen_representation = concatenate([left_sen_representation, right_sen_representation])
    print(sen_representation.shape)
    # senrep.append(sen_representation)
    # senrep=tf.stack(senrep)
    # sen_representation=tf.stack(sen_representation)
    # sen_representation=tf.reshape(sen_representation,shape=[-1,1,400])
    sen_representation=Lambda(lambda x: tf.reshape(sen_representation, shape=[-1,1,400]))(sen_representation)
    print(sen_representation.shape)
    avg_pool = GlobalAveragePooling1D()(sen_representation)
    max_pool = GlobalMaxPooling1D()(sen_representation)
    concat = concatenate([avg_pool, max_pool])
    dropout = Dropout(0.3)(concat)
    similarity = Dense(3, activation="softmax")(dropout)


    # 引入曼哈顿距离，把得到的变换concat上原始的向量再通过一个多层的DNN做了下非线性变换、sigmoid得相似度
    # 没有使用https://zhuanlan.zhihu.com/p/31638132中提到的马氏距离，尝试了曼哈顿距离、点乘和cos，效果曼哈顿最好

    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    # malstm_trained = model.fit( [ X_train['left'], X_train['right']], Y_train,
    #                            batch_size=batch_size, epochs=n_epoch,
    #                            validation_data=(
    #                            [X_validation['left'], X_validation['right']], Y_validation))

    malstm_trained = model.fit([BERT_train_question1, BERT_train_question2], y_train,
                               steps_per_epoch=batch_size, epochs=n_epoch,
                               validation_data=(
                                   [ BERT_test_question1, BERT_test_question2], y_val),validation_steps=1, validation_split=0)
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    # yhat_probs = model.predict([BERT_test_question1, BERT_test_question2], steps=1, verbose=0)
    # # predict crisp classes for test set
    # yhat_classes = np.argmax(yhat_probs, axis=1)
    # # yhat_classes = model.predict_classes([BERT_test_question1, BERT_test_question2], verbose=0)
    # yhat_probs = yhat_probs[:, 0]
    # # yhat_classes = yhat_classes[:, 0]
    # yhat_classes = yhat_probs > 0.5
    # accuracy = accuracy_score(Y_validation, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # # precision tp / (tp + fp)
    # precision = precision_score(Y_validation, yhat_classes)
    # print('Precision: %f' % precision)
    # # recall: tp / (tp + fn)
    # recall = recall_score(Y_validation, yhat_classes)
    # print('Recall: %f' % recall)
    # # f1: 2 tp / (2 tp + fp + fn)
    # f1 = f1_score(Y_validation, yhat_classes)
    # print('F1 score: %f' % f1)
    # Plot accuracy
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject/HHH-An-Online-Question-Answering-System-for-Medical-Questions/history-graphBERTBILSTM.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")


