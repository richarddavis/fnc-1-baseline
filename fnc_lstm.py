import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LSTM, GRU
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
from keras.utils.visualize_util import plot
from keras.engine.topology import Merge

MAX_NB_WORDS = 5000
max_words = 100
batch_size = 64

print('Loading data...')
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

print(len(X_headline), 'train headline sequences')
print(len(X_body), 'train body sequences')
print(len(y), 'train labels')
print(len(X_headline_test), 'test headline sequences')
print(len(X_body_test), 'test body sequences')
print(len(y_test), 'test labels')

# Transforms each text in texts in a sequence of integers.
# Only top "nb_words" most frequent words will be taken into account.
# Only words known by the tokenizer will be taken into account.
headline_sequences = tokenizer.texts_to_sequences(X_headline)
body_sequences = tokenizer.texts_to_sequences(X_body)
headline_sequences_test = tokenizer.texts_to_sequences(X_headline_test)
body_sequences_test = tokenizer.texts_to_sequences(X_body_test)

X_headline = sequence.pad_sequences(headline_sequences, maxlen=max_words)
X_body = sequence.pad_sequences(body_sequences, maxlen=max_words)
X_headline_test = sequence.pad_sequences(headline_sequences_test, maxlen=max_words)
X_body_test = sequence.pad_sequences(body_sequences_test, maxlen=max_words)

print(X_headline[0])
print(X_body[0])

print('X_headline shape:', X_headline.shape)
print('X_body shape:', X_body.shape)
print('X_headline_test shape:', X_headline_test.shape)
print('X_body_test shape:', X_body_test.shape)

y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)

print('Build model...')
headline_branch = Sequential()
headline_branch.add(Embedding(input_dim=MAX_NB_WORDS+2, output_dim=32, input_length=max_words, mask_zero=True))
headline_branch.add(GRU(output_dim=64))  # try using a GRU instead, for fun

body_branch = Sequential()
body_branch.add(Embedding(input_dim=MAX_NB_WORDS+2, output_dim=32, input_length=max_words, mask_zero=True))
body_branch.add(GRU(output_dim=64))  # try using a GRU instead, for fun

merged = Merge([headline_branch, body_branch], mode='sum')

model = Sequential()
model.add(merged)
model.add(Dense(400, activation='relu', init="glorot_normal"))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
plot(model, to_file='fnc_lstm.png', show_shapes=True)

print('Train...')
model.fit([X_headline, X_body], y, validation_data=([X_headline_test, X_body_test], y_test), nb_epoch=4, batch_size=batch_size, verbose=1)
# model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
#           validation_data=(X_test, y_test))
# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
predicted = model.predict([X_headline_test, X_body_test])
report_score([LABELS[np.where(x==1)[0][0]] for x in y_test],
             [LABELS[np.argmax(x)] for x in predicted])
