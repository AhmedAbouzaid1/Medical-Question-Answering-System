from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import numpy as np
from gensim.models import KeyedVectors
from keras import initializers as initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D,GlobalAveragePooling1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K



from util import make_w2v_embeddings, split_and_zero_padding, ManDist
# from BertEncoder import BertSentenceEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
'''pip
本配置文件用于训练孪生网络
'''

# ------------------预加载------------------ #

TRAIN_CSV = 'C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject1/SNLI_Corpus/snli_1.0_train.csv'

flag = 'en'
embedding_path = 'wiki-news-300d-1M.vec'
embedding_dim = 300
max_seq_length = 10
savepath = 'C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject1/simmodel_FastText.h5'

train_df = pd.read_csv(TRAIN_CSV, encoding='gb18030')





print("Loading word2vec model(it may takes 2-3 mins) ...")
embedding_dict = KeyedVectors.load_word2vec_format(embedding_path)


#
# print(train_df['question1'])
for q in ['sentence1', 'sentence2']:
    train_df[q + '_n'] = train_df[q]



print(train_df['sentence1_n'], train_df['sentence2_n'])
print(train_df.head())
train_df = train_df[0:12000]

train_df, embeddings = make_w2v_embeddings(flag, embedding_dict, train_df, embedding_dim=embedding_dim)




# Y_train = train_df['similarity'][0:10000]
# Y_validation = train_df['similarity'][10000:11000]
# Y_testing = train_df['similarity'][11000:12000]


Y_train = train_df["similarity"][0:10000].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
Y_train=keras.utils.to_categorical(Y_train, num_classes=3)

Y_validation = train_df["similarity"][10000:11000].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
Y_validation=keras.utils.to_categorical(Y_validation, num_classes=3)

Y_testing = train_df["similarity"][11000:12000].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
Y_testing=keras.utils.to_categorical(Y_testing, num_classes=3)

X_train= train_df[['sentence1_n', 'sentence2_n']][0:10000]
X_validation= train_df[['sentence1_n', 'sentence2_n']][10000:11000]
X_testing= train_df[['sentence1_n', 'sentence2_n']][11000:12000]


X_train = split_and_zero_padding(X_train, max_seq_length)  #this function renames the columns to left and right
X_validation = split_and_zero_padding(X_validation, max_seq_length)
X_testing = split_and_zero_padding(X_testing, max_seq_length)

# 将标签转化为数值
# Y_train = Y_train.values
# Y_validation = Y_validation.values

#concatenating q1 an q2 after embeddings

# X_train['con_q']= X_train['left']+X_train['right']
# X_validation['con_q']= X_validation['left']+X_validation['right']
# X_testing['con_q']= X_testing['left']+X_testing['right']
# print("con_q", X_testing['con_q'].shape)
train_con=[]
for i in range(0,10000):
    # print("con_q", X_train['left'].shape)
    y = np.append (X_train['left'][i],X_train['right'][i] )
    # print("y",y)
    # print("y", y.shape)
    train_con.append(y)

print("lolo", np.shape(train_con))
X_train['con_q']=train_con
X_train['con_q']=np.asarray(X_train['con_q'])
# print("shape", X_train['con_q'].shape)

valid_con=[]
for i in range(0,1000):
    # print("con_q", X_validation['left'].shape)
    y = np.append (X_validation['left'][i],X_validation['right'][i] )
    # print("y",y)
    # print("y", y.shape)
    valid_con.append(y)

X_validation['con_q']=valid_con
X_validation['con_q']=np.asarray(X_validation['con_q'])


test_con=[]
for i in range(0,1000):
    # print("con_q", X_validation['left'].shape)
    y = np.append (X_testing['left'][i],X_testing['right'][i] )
    # print("y",y)
    # print("y", y.shape)
    test_con.append(y)

X_testing['con_q']=test_con
X_testing['con_q']=np.asarray(X_testing['con_q'])

# 确认数据准备完毕且正确
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

def sim_model(_input):


    embedding_layer = Embedding(len(embeddings) + 1,
                                embedding_dim,
                                input_length=max_seq_length*2)

    print(embedding_layer)
    print(type(embedding_layer))

    embedded_sequences = embedding_layer(_input)
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True)
    )(embedded_sequences)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = GlobalAveragePooling1D()(bi_lstm)
    max_pool = GlobalMaxPool1D()(bi_lstm)
    concat = concatenate([avg_pool, max_pool])
    dropout = Dropout(0.3)(concat)
    output = Dense(3, activation="softmax")(dropout)



    return output



if __name__ == '__main__':
    # 超参
    batch_size = 1024
    n_epoch = 20
    n_hidden = 50
    # left_input = Input(shape=(1,), dtype='float32')
    _input = Input(shape=(max_seq_length*2,), dtype='float32')
    print('this is left inout', _input)
    similarity = sim_model(_input)

    model = Model(inputs=[_input], outputs=[similarity])

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    training_start_time = time()
    print("inout shape",X_train['con_q'].shape)
    print("y shape", np.shape(Y_train))

    _trained = model.fit(X_train['con_q'], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=(
                                   X_validation['con_q'], Y_validation))
    training_end_time = time()

    model.evaluate(X_testing['con_q'],Y_testing, verbose=1)

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(_trained.history['accuracy'])
    plt.plot(_trained.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(_trained.history['loss'])
    plt.plot(_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('C:/Users/CSE-P07-2179-G9/PycharmProjects/pythonProject1/simmodel_FastText.png')

    model.save(savepath)
    print(str(_trained.history['val_accuracy'][-1])[:6] +
          "(max: " + str(max(_trained.history['val_accuracy']))[:6] + ")")
    print("Done.")


