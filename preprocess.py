import os
import numpy as np

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils.dataset import DataSet
import feature_engineering as fe

MAX_NB_WORDS = None
MAX_SEQUENCE_LENGTH = 200
GLOVE_DIR = './wordvectors'
EMBEDDING_DIM = 100

d = DataSet()

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
body_text = [v for k,v in d.articles.items()]
headline_text = [s['Headline'] for s in d.stances]

tokenizer.fit_on_texts(body_text + headline_text)

headline_sequences = tokenizer.texts_to_sequences(headline_text)
body_sequences = tokenizer.texts_to_sequences(body_text)

word_index = tokenizer.word_index # word_index is our vocabulary: a dictionary linking tokens (string) to IDs.
print('Found %s unique tokens.' % len(word_index))

# Now let's create the embedding layer

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

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
