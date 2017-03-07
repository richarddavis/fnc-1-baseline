import os
import numpy as np

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils.dataset import DataSet
from utils.generate_test_splits import generate_hold_out_split, read_ids
from utils.nn import generate_ff_features

MAX_NB_WORDS = None
MAX_SEQUENCE_LENGTH = 200
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
# Part 5: Create the embedding layer
# ----------------------------------------------------

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
