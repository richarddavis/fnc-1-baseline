# MLP for the IMDB problem
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Flatten, Merge
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
from sklearn.metrics import classification_report, confusion_matrix

VOCAB_SIZE = 3000
max_seq_len = 200
epochs = 14

# Load the dataset using the utility provided by the FNC
d = DataSet()

# Create Keras tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)

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

# X_headline = sequence.pad_sequences(headline_sequences, maxlen=max_seq_len)
# X_body = sequence.pad_sequences(body_sequences, maxlen=max_seq_len)
# X_headline_test = sequence.pad_sequences(headline_sequences_test, maxlen=max_seq_len)
# X_body_test = sequence.pad_sequences(body_sequences_test, maxlen=max_seq_len)

X_headline = tokenizer.sequences_to_matrix(headline_sequences, mode='binary')
X_body = tokenizer.sequences_to_matrix(body_sequences, mode='binary')
X_headline_test = tokenizer.sequences_to_matrix(headline_sequences_test, mode='binary')
X_body_test = tokenizer.sequences_to_matrix(body_sequences_test, mode='binary')

y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)

# create the headline and body input branches
left_input = Input(shape=(VOCAB_SIZE,), dtype='float32')
right_input = Input(shape=(VOCAB_SIZE,), dtype='float32')
# left_branch = Dense(150, activation='relu', init='glorot_normal')(left_input)
# right_branch = Dense(150, activation='relu', init='glorot_normal')(right_input)

# merge the headline and body input branches
merged = keras.layers.concatenate([left_input, right_input])
merged = Dense(600, activation='relu', init="glorot_normal")(merged)
# merged = Dense(300, activation='relu', init="glorot_normal")(merged)
merged = Dropout(0.6)(merged)
out = Dense(4, activation='softmax')(merged)
model = Model(inputs=[left_input, right_input], output=out)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='ff_sequence_matrix.png', show_shapes=True)

# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochsh=2, batch_size=128, verbose=1)
# model.fit([X_headline, X_body], y, epochsh=5, batch_size=128, verbose=1)
model.fit([X_headline, X_body], y, validation_data=([X_headline_test, X_body_test], y_test), epochs=epochs, batch_size=128, verbose=1)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

predicted = model.predict([X_headline_test, X_body_test])
report_score([LABELS[np.where(x==1)[0][0]] for x in y_test],
             [LABELS[np.argmax(x)] for x in predicted])
