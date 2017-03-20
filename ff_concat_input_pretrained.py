# MLP for the IMDB problem
import os
import numpy as np
import keras
from keras.datasets import imdb
from keras.layers import Input, LSTM, Dense, merge, Flatten, concatenate
from keras.models import Sequential
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from utils.dataset import DataSet
from utils.generate_data import generate_data
from utils.generate_test_splits import generate_hold_out_split, read_ids
from utils.score import report_score, LABELS
from keras.utils import np_utils
from keras.utils import plot_model

GLOVE_DIR = './wordvectors'
EMBEDDING_DIM = 50
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100

# Load the dataset using the utility provided by the FNC
d = DataSet()

# Create Keras tokenizer
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

# Collect all text from bodies and headlines of data provided for FNC
body_text = [v for k,v in d.articles.items()]
headline_text = [s['Headline'] for s in d.stances]

# Updates internal vocabulary based on a list of texts
tokenizer.fit_on_texts(headline_text + body_text)

# Create lists of headline text, body text, and stances

generate_hold_out_split(d)
base_dir = "splits"
training_ids = read_ids("training_ids.txt", base_dir)
hold_out_ids = read_ids("hold_out_ids.txt", base_dir)

X_headline, X_body, y = generate_data(training_ids, d)
X_headline_test, X_body_test, y_test = generate_data(hold_out_ids, d)
# Create sequences from the lists of texts

# Transforms each text in texts in a sequence of integers.
# Only top "nb_words" most frequent words will be taken into account.
# Only words known by the tokenizer will be taken into account.
headline_sequences = tokenizer.texts_to_sequences(X_headline)
body_sequences = tokenizer.texts_to_sequences(X_body)
headline_sequences_test = tokenizer.texts_to_sequences(X_headline_test)
body_sequences_test = tokenizer.texts_to_sequences(X_body_test)
word_index = tokenizer.word_index

X_headline = sequence.pad_sequences(headline_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_body = sequence.pad_sequences(body_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_headline_test = sequence.pad_sequences(headline_sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
X_body_test = sequence.pad_sequences(body_sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)

# Create pre-trained embedding matrix
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


headline_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
body_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
left_branch = embedding_layer(headline_input)
print (left_branch.shape)
left_branch = Flatten()(left_branch)
print (left_branch.shape)
right_branch = embedding_layer(body_input)
right_branch = Flatten()(right_branch)
merged_vector = concatenate([left_branch, right_branch])
x = Dense(200, activation='relu')(merged_vector)
x = Dropout(0.5)(x)
preds = Dense(4, activation='softmax')(x)
model = Model(input=[headline_input, body_input], output=preds)

model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='ff_concat_input_pretrained.png', show_shapes=True)
model.fit([X_headline, X_body], y, nb_epoch=5, batch_size=64)

# # create the model
# left_branch = Sequential()
# # left_branch.add(Embedding(MAX_NB_WORDS, 20, input_length=MAX_SEQUENCE_LENGTH))
# left_branch.add(embedding_layer)
# left_branch.add(Flatten())

# right_branch = Sequential()
# # right_branch.add(Embedding(MAX_NB_WORDS, 20, input_length=MAX_SEQUENCE_LENGTH))
# right_branch.add(embedding_layer)
# right_branch.add(Flatten())

# merged = Merge([left_branch, right_branch], mode='concat')

# model = Sequential()
# model.add(merged)
# model.add(Dense(200, activation='relu', W_regularizer=l2(0.001), init="glorot_normal"))
# model.add(Dropout(0.5))
# model.add(Dense(200, activation='relu', W_regularizer=l2(0.001), init="glorot_normal"))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
# print(model.summary())

# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
# model.fit([X_headline, X_body], y, nb_epoch=5, batch_size=64, verbose=1)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

predicted = model.predict([X_headline_test, X_body_test])
report_score([LABELS[np.where(x==1)[0][0]] for x in y_test],
             [LABELS[np.argmax(x)] for x in predicted])
