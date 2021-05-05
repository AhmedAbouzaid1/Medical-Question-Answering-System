from elmo import ELMoEmbedding
ELMoEmbedding(idx2word="idx2word", output_mode="default", trainable=True)




# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import transformers
# import os
#
# import pandas as pd
# import tensorflow_hub as hub
# import os
# import re
# # from keras import backend as K
#
# import keras.layers as layers
# from keras.models import Model, load_model
# from keras.engine import Layer
# import numpy as np
# from tensorflow import keras
#
# from tensorflow.python.keras.callbacks import ModelCheckpoint
#
# # Set the config values
#
#
# # Initialize session
# from tensorflow.python.keras import backend as K
# sess = tf.compat.v1.Session()
#
# K.set_session(sess)
#
# max_length = 128  # Maximum length of input sentence to the model.
# batch_size = 32
# epochs = 2
#
# def load_directory_data(directory):
#   data = {}
#   data["sentence"] = []
#   data["sentiment"] = []
#   for file_path in os.listdir(directory):
#     with tf.compat.v1.gfile.GFile(os.path.join(directory, file_path), "r") as f:
#       data["sentence"].append(f.read())
#       data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
#   return pd.DataFrame.from_dict(data)
#
# # Merge positive and negative examples, add a polarity column and shuffle.
# def load_dataset(directory):
#   pos_df = load_directory_data(os.path.join(directory, "pos"))
#   neg_df = load_directory_data(os.path.join(directory, "neg"))
#   pos_df["polarity"] = 1
#   neg_df["polarity"] = 0
#   return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
#
# # Download and process the dataset files.
# def download_and_load_datasets(force_download=False):
#   dataset = tf.keras.utils.get_file(
#       fname="aclImdb.tar.gz",
#       origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
#       extract=True)
#
#   train_df = load_dataset(os.path.join(os.path.dirname(dataset),
#                                        "aclImdb", "train"))
#   test_df = load_dataset(os.path.join(os.path.dirname(dataset),
#                                       "aclImdb", "test"))
#
#   return train_df, test_df
#
# # Reduce logging output.
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#
# train_df, test_df = download_and_load_datasets()
# train_df.head()
#
# class ElmoEmbeddingLayer(Layer):
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
#
#
# def build_model():
#     input_text = layers.Input(shape=(1,), dtype="string")
#     embedding = ElmoEmbeddingLayer()(input_text)
#     dense = layers.Dense(256, activation='relu')(embedding)
#     pred = layers.Dense(1, activation='sigmoid')(dense)
#
#     model = Model(inputs=[input_text], outputs=pred)
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#
#     return model
#
# train_text = train_df['sentence'].tolist()
# train_text = [' '.join(t.split()[0:150]) for t in train_text]
# train_text = np.array(train_text, dtype=object)[:, np.newaxis]
# train_label = train_df['polarity'].tolist()
#
# test_text = test_df['sentence'].tolist()
# test_text = [' '.join(t.split()[0:150]) for t in test_text]
# test_text = np.array(test_text, dtype=object)[:, np.newaxis]
# test_label = test_df['polarity'].tolist()
#
# model = build_model()
# model.fit(train_text,
#           train_label,
#           validation_data=(test_text, test_label),
#           epochs=1,
#           batch_size=32)
#
# model.save('ElmoModel.h5')
# pre_save_preds = model.predict(test_text[0:100]) # predictions before we clear and reload model
#
# # Clear and load model
# model = None
# model = build_model()
# model.load_weights('ElmoModel.h5')
#
# post_save_preds = model.predict(test_text[0:100]) # predictions after we clear and reload model
# all(pre_save_preds == post_save_preds) # Are they the same?
# #
# # # Labels in our dataset.
# # labels = ["contradiction", "entailment", "neutral"]
# #
# #
# # # There are more than 550k samples in total; we will use 100k for this example.
# # train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
# # valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
# # test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")
# # train_df = train_df[:10000]
# # valid_df = valid_df[:10000]
# # test_df = test_df[:1000]
# # # Shape of the data
# # print(f"Total train samples : {train_df.shape[0]}")
# # print(f"Total validation samples: {valid_df.shape[0]}")
# # print(f"Total test samples: {valid_df.shape[0]}")
# #
# # print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
# # print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
# # print(f"Similarity: {train_df.loc[1, 'similarity']}")
# #
# # # We have some NaN entries in our train data, we will simply drop them.
# # print("Number of missing values")
# # print(train_df.isnull().sum())
# # train_df.dropna(axis=0, inplace=True)
# #
# # print("Train Target Distribution")
# # print(train_df.similarity.value_counts())
# #
# #
# # print("Validation Target Distribution")
# # print(valid_df.similarity.value_counts())
# #
# # train_df = (
# #     train_df[train_df.similarity != "-"]
# #     .sample(frac=1.0, random_state=42)
# #     .reset_index(drop=True)
# # )
# # valid_df = (
# #     valid_df[valid_df.similarity != "-"]
# #     .sample(frac=1.0, random_state=42)
# #     .reset_index(drop=True)
# # )
# #
# # train_df["label"] = train_df["similarity"].apply(
# #     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
# # )
# # y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)
# #
# # valid_df["label"] = valid_df["similarity"].apply(
# #     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
# # )
# # y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)
# #
# # test_df["label"] = test_df["similarity"].apply(
# #     lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
# # )
# # y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)
# #
# #
# # class BertSemanticDataGenerator(tf.keras.utils.Sequence):
# #     """Generates batches of data.
# #
# #     Args:
# #         sentence_pairs: Array of premise and hypothesis input sentences.
# #         labels: Array of labels.
# #         batch_size: Integer batch size.
# #         shuffle: boolean, whether to shuffle the data.
# #         include_targets: boolean, whether to incude the labels.
# #
# #     Returns:
# #         Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
# #         (or just `[input_ids, attention_mask, `token_type_ids]`
# #          if `include_targets=False`)
# #     """
# #
# #     def __init__(
# #         self,
# #         sentence_pairs,
# #         labels,
# #         batch_size=batch_size,
# #         shuffle=True,
# #         include_targets=True,
# #     ):
# #         self.sentence_pairs = sentence_pairs
# #         self.labels = labels
# #         self.shuffle = shuffle
# #         self.batch_size = batch_size
# #         self.include_targets = include_targets
# #         # Load our BERT Tokenizer to encode the text.
# #         # We will use base-base-uncased pretrained model.
# #         self.tokenizer = transformers.BertTokenizer.from_pretrained(
# #             "bert-base-uncased", do_lower_case=True
# #         )
# #         self.indexes = np.arange(len(self.sentence_pairs))
# #         self.on_epoch_end()
# #
# #     def __len__(self):
# #         # Denotes the number of batches per epoch.
# #         return len(self.sentence_pairs) // self.batch_size
# #
# #     def __getitem__(self, idx):
# #         # Retrieves the batch of index.
# #         indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
# #         sentence_pairs = self.sentence_pairs[indexes]
# #
# #         # With BERT tokenizer's batch_encode_plus batch of both the sentences are
# #         # encoded together and separated by [SEP] token.
# #         encoded = self.tokenizer.batch_encode_plus(
# #             sentence_pairs.tolist(),
# #             add_special_tokens=True,
# #             max_length=max_length,
# #             return_attention_mask=True,
# #             return_token_type_ids=True,
# #             pad_to_max_length=True,
# #             return_tensors="tf",
# #         )
# #
# #         # Convert batch of encoded features to numpy array.
# #         input_ids = np.array(encoded["input_ids"], dtype="int32")
# #         attention_masks = np.array(encoded["attention_mask"], dtype="int32")
# #         token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
# #
# #         # Set to true if data generator is used for training/validation.
# #         if self.include_targets:
# #             labels = np.array(self.labels[indexes], dtype="int32")
# #             return [input_ids, attention_masks, token_type_ids], labels
# #         else:
# #             return [input_ids, attention_masks, token_type_ids]
# #
# #     def on_epoch_end(self):
# #         # Shuffle indexes after each epoch if shuffle is set to True.
# #         if self.shuffle:
# #             np.random.RandomState(42).shuffle(self.indexes)
# #
# #
# # # Create the model under a distribution strategy scope.
# # strategy = tf.distribute.MirroredStrategy()
# #
# # with strategy.scope():
# #     # Encoded token ids from BERT tokenizer.
# #     input_ids = tf.keras.layers.Input(
# #         shape=(max_length,), dtype=tf.int32, name="input_ids"
# #     )
# #     # Attention masks indicates to the model which tokens should be attended to.
# #     attention_masks = tf.keras.layers.Input(
# #         shape=(max_length,), dtype=tf.int32, name="attention_masks"
# #     )
# #     # Token type ids are binary masks identifying different sequences in the model.
# #     token_type_ids = tf.keras.layers.Input(
# #         shape=(max_length,), dtype=tf.int32, name="token_type_ids"
# #     )
# #     # Loading pretrained BERT model.
# #     from transformers import BertModel, TFBertModel
# #     bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
# #     # Freeze the BERT model to reuse the pretrained features without modifying them.
# #     bert_model.trainable = False
# #
# #     sequence_output, pooled_output = bert_model.bert(
# #         input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
# #     )
# #     # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
# #     biGRU = tf.keras.layers.Bidirectional(
# #         tf.keras.layers.GRU(64, return_sequences=True)
# #     )(sequence_output)
# #     bilstm = tf.keras.layers.Bidirectional(
# #         tf.keras.layers.LSTM(64, return_sequences=True)
# #     )(sequence_output)
# #     # CNN = tf.keras.layers.Conv1D(filters=32, kernel_size= 5, activation='relu')(sequence_output)
# #     # avg_pool = tf.keras.layers.AveragePooling1D(3)(CNN)
# #     # CNN = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(avg_pool)
# #     # max_pool = tf.keras.layers.MaxPooling1D(3)(CNN)
# #     # CNN = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(max_pool)
# #     # avg_pool = tf.keras.layers.GlobalAveragePooling1D()(CNN)
# #     # max_pool = tf.keras.layers.GlobalMaxPooling1D()(CNN)
# #     # concat = tf.keras.layers.concatenate([avg_pool, max_pool])
# #     # Applying hybrid pooling approach to biGRU sequence output.
# #     avg_pool = tf.keras.layers.GlobalAveragePooling1D()(biGRU)
# #     max_pool = tf.keras.layers.GlobalMaxPooling1D()(biGRU)
# #     avg_pool1 = tf.keras.layers.GlobalAveragePooling1D()(bilstm)
# #     max_pool1 = tf.keras.layers.GlobalMaxPooling1D()(bilstm)
# #     concat = tf.keras.layers.concatenate([avg_pool, max_pool])
# #     concat1 = tf.keras.layers.concatenate([avg_pool1, max_pool1])
# #     concat_all = tf.keras.layers.concatenate([concat, concat1])
# #
# #     dropout = tf.keras.layers.Dropout(0.3)(concat_all)
# #     output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
# #     model = tf.keras.models.Model(
# #         inputs=[input_ids, attention_masks, token_type_ids], outputs=output
# #     )
# #
# #     model.compile(
# #         optimizer=tf.keras.optimizers.Adam(),
# #         loss="categorical_crossentropy",
# #         metrics=["acc"],
# #     )
# #
# #
# # print(f"Strategy: {strategy}")
# # model.summary()
# #
# # train_data = BertSemanticDataGenerator(
# #     train_df[["sentence1", "sentence2"]].values.astype("str"),
# #     y_train,
# #     batch_size=batch_size,
# #     shuffle=True,
# # )
# # valid_data = BertSemanticDataGenerator(
# #     valid_df[["sentence1", "sentence2"]].values.astype("str"),
# #     y_val,
# #     batch_size=batch_size,
# #     shuffle=False,
# # )
# #
# # history = model.fit(
# #     train_data,
# #     validation_data=valid_data,
# #     epochs=epochs,
# #     use_multiprocessing=True,
# #     workers=-1,
# # )
# #
# #
# #
# # # Unfreeze the bert_model.
# # bert_model.trainable = True
# # # Recompile the model to make the change effective.
# # model.compile(
# #     optimizer=tf.keras.optimizers.Adam(1e-5),
# #     loss="categorical_crossentropy",
# #     metrics=["accuracy"],
# # )
# # model.summary()
# #
# # history = model.fit(
# #     train_data,
# #     validation_data=valid_data,
# #     epochs=epochs,
# #     use_multiprocessing=True,
# #     workers=-1,
# # )
# #
# # test_data = BertSemanticDataGenerator(
# #     test_df[["sentence1", "sentence2"]].values.astype("str"),
# #     y_test,
# #     batch_size=batch_size,
# #     shuffle=False,
# # )
# # model.evaluate(test_data, verbose=1)
# #
# #
# # def check_similarity(sentence1, sentence2):
# #     sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
# #     test_data = BertSemanticDataGenerator(
# #         sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
# #     )
# #
# #     proba = model.predict(test_data)[0]
# #     idx = np.argmax(proba)
# #     proba = f"{proba[idx]: .2f}%"
# #     pred = labels[idx]
# #     return pred, proba
# # checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# # checkpoint_dir = os.path.dirname(checkpoint_path)
# #
# # batch_size = 32
# #
# #
# # # Create a callback that saves the model's weights every 5 epochs
# # cp_callback = tf.keras.callbacks.ModelCheckpoint(
# #     filepath=checkpoint_path,
# #     verbose=1,
# #     save_weights_only=True,
# #     save_freq=5*batch_size)
# #
# # model.save_weights(checkpoint_path.format(epoch=0))
# # # model.save('saved_model1/my_model')
# # model.save('model_bigru(max_10k).h5')
# #
