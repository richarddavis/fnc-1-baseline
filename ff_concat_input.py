# MLP for the Fake News Challenge
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
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

MAX_NB_WORDS = 10000
max_words = 500

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

X_headline = sequence.pad_sequences(headline_sequences, maxlen=max_words)
X_body = sequence.pad_sequences(body_sequences, maxlen=max_words)
X_headline_test = sequence.pad_sequences(headline_sequences_test, maxlen=max_words)
X_body_test = sequence.pad_sequences(body_sequences_test, maxlen=max_words)

y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)

# create the model
left_branch = Sequential()
left_branch.add(Embedding(MAX_NB_WORDS, 20, input_length=max_words))
left_branch.add(Flatten())

right_branch = Sequential()
right_branch.add(Embedding(MAX_NB_WORDS, 20, input_length=max_words))
right_branch.add(Flatten())

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(200, activation='relu', W_regularizer=l2(0.001), init="glorot_normal"))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu', W_regularizer=l2(0.001), init="glorot_normal"))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
print(model.summary())
plot(model, to_file='ff_concat_input.png', show_shapes=True)

# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
model.fit([X_headline, X_body], y, nb_epoch=5, batch_size=64, verbose=1)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

predicted = model.predict([X_headline_test, X_body_test])
report_score([LABELS[np.where(x==1)[0][0]] for x in y_test],
             [LABELS[np.argmax(x)] for x in predicted])
