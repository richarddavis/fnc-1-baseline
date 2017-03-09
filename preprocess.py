import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop, Nadam
from keras.layers import Dense

from utils.dataset import DataSet
from utils.generate_test_splits import generate_hold_out_split, read_ids
from utils.nn import generate_ff_features
from utils.score import report_score, LABELS

MAX_NB_WORDS = None
MAX_SEQUENCE_LENGTH = 1000
GLOVE_DIR = './wordvectors'
EMBEDDING_DIM = 100

# ----------------------------------------------------
# Part 1: Load the Dataset.
# ----------------------------------------------------

# Load the dataset using the utility provided by the FNC
d = DataSet()

# ----------------------------------------------------
# Part 2: Tokenize the texts and create the vocabulary
# ----------------------------------------------------

# Create Keras tokenizer
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

# Collect all text from bodies and headlines of data provided for FNC
body_text = [v for k,v in d.articles.items()]
headline_text = [s['Headline'] for s in d.stances]

# Updates internal vocabulary based on a list of texts
tokenizer.fit_on_texts(body_text + headline_text)

# Transforms each text in texts in a sequence of integers.
# Only top "nb_words" most frequent words will be taken into account.
# Only words known by the tokenizer will be taken into account.
headline_sequences = tokenizer.texts_to_sequences(headline_text)
body_sequences = tokenizer.texts_to_sequences(body_text)

# word_index is a dictionary mapping words (str) to their rank/index (int)
# The index will be the word vector's location in the embedding matrix
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# ----------------------------------------------------
# Part 3: Create the embedding matrix
# ----------------------------------------------------

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
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


# ----------------------------------------------------
# Part 4: Create the training and test data
# ----------------------------------------------------

# This model is basic
# 1. The word vectors for each word in the headline are summed to create a single headline vector.
# 2. The word vectors for each word in the article are summed to create a single article vector.

# (Didn't do this, but maybe I should)
# Create a lookup vector where each word (represented as a one-hot vector) is summed. When this is
# multiplied by the embedding matrix the result will be the sum of all the word vectors.

# Use utilities to create training and validation sets.
generate_hold_out_split(d)
base_dir = "splits"
training_ids = read_ids("training_ids.txt", base_dir)
hold_out_ids = read_ids("hold_out_ids.txt", base_dir)

# Generate the data. Each row of X is the summed headline vector concatenated with the summed body vector.
X, y = generate_ff_features(training_ids, d, embedding_matrix, tokenizer, EMBEDDING_DIM)

# ----------------------------------------------------
# Part 5: Encode the labels, convert from ints to one-hot vectors
# ----------------------------------------------------

# le = LabelEncoder()
# labels = le.fit_transform(y)

# Transform the labels into vectors in the range [0, num_classes].
# This generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`

labels = np_utils.to_categorical(y)

# ----------------------------------------------------
# Part 6: Use sklearn to split into train and test sets
# ----------------------------------------------------

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	X, labels, test_size=0.25, random_state=42)

# ----------------------------------------------------
# Part 7: Create the model!
# ----------------------------------------------------

# define the architecture of the network
model = Sequential()
model.add(Dense(100, input_dim=EMBEDDING_DIM*2, init="glorot_normal",
	activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation("softmax"))

# ----------------------------------------------------
# Part 8: Train the Model
# ----------------------------------------------------

print("[INFO] compiling model...")
sgd = SGD(lr=0.000001)
nadam = Nadam()
adam = Adam(lr=0.00001)
# model.compile(loss="categorical_crossentropy", optimizer=adam,
# 	metrics=["categorical_accuracy"])
model.compile(loss="binary_crossentropy", optimizer=adam,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=20, batch_size=32,
	verbose=1)

# ----------------------------------------------------
# Part 9: Show accuracy on the test set
# ----------------------------------------------------

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

# Use the provided functionality to determine the confusion matrix and accuracy
predicted = model.predict(testData)
report_score([LABELS[np.where(x==1)[0][0]] for x in testLabels],
             [LABELS[np.argmax(x)] for x in predicted])

# ----------------------------------------------------
# Create the embedding layer
# ----------------------------------------------------

# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)
