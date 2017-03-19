import numpy as np
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Embedding, Flatten, LSTM, GRU, concatenate
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

MAX_HEADLINE_LENGTH = 100
MAX_BODY_LENGTH = 1000
EMBEDDING_DIM = 20
VOCAB_SIZE = 3000
BATCH_SIZE = 64
NUM_EPOCHS = 4

print('Loading data...')
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

X_headline = sequence.pad_sequences(headline_sequences, maxlen=MAX_HEADLINE_LENGTH)
X_body = sequence.pad_sequences(body_sequences, maxlen=MAX_BODY_LENGTH)
X_headline_test = sequence.pad_sequences(headline_sequences_test, maxlen=MAX_HEADLINE_LENGTH)
X_body_test = sequence.pad_sequences(body_sequences_test, maxlen=MAX_BODY_LENGTH)

print(X_headline[0])
print(X_body[0])

print('X_headline shape:', X_headline.shape)
print('X_body shape:', X_body.shape)
print('X_headline_test shape:', X_headline_test.shape)
print('X_body_test shape:', X_body_test.shape)

y = np_utils.to_categorical(y)
y_test = np_utils.to_categorical(y_test)

print('Build model...')
headline_input = Input(shape=(MAX_HEADLINE_LENGTH,), dtype='int32')
body_input = Input(shape=(MAX_BODY_LENGTH,), dtype='int32')
headline_branch = Embedding(input_dim=VOCAB_SIZE+2, output_dim=20, input_length=MAX_HEADLINE_LENGTH, mask_zero=True)(headline_input)
body_branch = Embedding(input_dim=VOCAB_SIZE+2, output_dim=20, input_length=MAX_BODY_LENGTH, mask_zero=True)(body_input)
headline_branch = GRU(output_dim=128, go_backwards=True)(headline_branch)
body_branch = GRU(output_dim=128, go_backwards=True)(body_branch)

merged = concatenate([headline_branch, body_branch])
merged = Dense(400, activation='relu', init='glorot_normal')(merged)
merged = Dropout(0.2)(merged)
out = Dense(4, activation='softmax')(merged)
model = Model(inputs=[headline_input, body_input], output=out)

def fnc_accuracy(y_true, y_pred):
    pass

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='fnc_lstm.png', show_shapes=True)

print('Train...')
model.fit([X_headline, X_body], y, validation_data=([X_headline_test, X_body_test], y_test), nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
# model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
#           validation_data=(X_test, y_test))
# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
predicted = model.predict([X_headline_test, X_body_test])
report_score([LABELS[np.where(x==1)[0][0]] for x in y_test],
             [LABELS[np.argmax(x)] for x in predicted])

