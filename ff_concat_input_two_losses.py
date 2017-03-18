# MLP for the IMDB problem
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.layers import Flatten
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model

from keras.regularizers import l2

from utils.dataset import DataSet
from utils.generate_data import generate_data, collapse_stances
from utils.generate_test_splits import generate_hold_out_split, read_ids
from utils.score import report_score, LABELS
from keras.utils import np_utils

# ===========================================
# Hyperparameters
# ===========================================
VOCABULARY_DIM = 3000
ARTICLE_LENGTH = 500       
HEADLINE_LENGTH = 70 
ARTICLE_EMBEDDING_DIM = 20
HEADLINE_EMBEDDING_DIM = 20

HIDDEN_LAYERS = [       
    (200, 'relu', 0.5),
    (200, 'relu', 0.5),
]

COMPILE = {
    'optimizer': 'nadam', 
    'loss': {
        'related_prediction': 'binary_crossentropy',
        'stance_prediction': 'categorical_crossentropy'
    },
    'loss_weights': {
        'related_prediction': 0.25,
        'stance_prediction': 0.75
    },
    'metrics': ['accuracy']
}

FIT = {
    'epochs': 1,
    'batch_size': 64,
    'verbose': 1
}



# ===========================================
# Generate the data
# ===========================================

# Load the dataset using the utility provided by the FNC
d = DataSet()

# Create lists of headline text, article text, and stances
generate_hold_out_split(d)
base_dir = "splits"
training_ids = read_ids("training_ids.txt", base_dir)
X_headline, X_article, y_stance = generate_data(training_ids, d)
y_related = collapse_stances(y_stance)

# Flatten out integers to one-hots
y_stance = np_utils.to_categorical(y_stance)
y_related = np_utils.to_categorical(y_related)

# Create Keras tokenizer
# Note: The tokenizer should only be fit on training data.
tokenizer = Tokenizer(num_words=VOCABULARY_DIM)
tokenizer.fit_on_texts(X_headline + X_article)

# Transforms each text in texts in a sequence of integers.
# Only top "nb_words" most frequent words will be taken into account.
# Only words known by the tokenizer will be taken into account.
# ALTERNATIVELY, WE COULD USE A RNN TO SQUISH SEQUENCES INTO VECTORS.
headline_sequences = tokenizer.texts_to_sequences(X_headline)
article_sequences = tokenizer.texts_to_sequences(X_article)

X_headline = sequence.pad_sequences(headline_sequences, maxlen=HEADLINE_LENGTH)
X_article = sequence.pad_sequences(article_sequences, maxlen=ARTICLE_LENGTH)


# Pull the test data
hold_out_ids = read_ids("hold_out_ids.txt", base_dir)
X_test_headline, X_test_article, y_test_stance = generate_data(hold_out_ids, d)
y_test_related = collapse_stances(y_test_stance)
y_test_stance = np_utils.to_categorical(y_test_stance)
y_test_related = np_utils.to_categorical(y_test_related)

# sequence and pad
headline_sequences_test = tokenizer.texts_to_sequences(X_test_headline)
article_sequences_test = tokenizer.texts_to_sequences(X_test_article)
X_test_headline = sequence.pad_sequences(headline_sequences_test, maxlen=HEADLINE_LENGTH)
X_test_article = sequence.pad_sequences(article_sequences_test, maxlen=ARTICLE_LENGTH)

# ===========================================
# Build the model
# ===========================================

# Variants to try: 
    # Restrict headlines and articles to the same sequence length, use just one embedding matrix

# create the model
headline_input = Input(
    shape=(HEADLINE_LENGTH,), 
    dtype='int32', 
    name='headline_input'
)

headline_word_vectors = Embedding(
    input_dim=VOCABULARY_DIM, 
    input_length=HEADLINE_LENGTH,
    output_dim=HEADLINE_EMBEDDING_DIM,
    name='headline_embedding'
)(headline_input) 

print("BEFORE EMBEDDING")
print(headline_input.shape)
print("AFTER EMBEDDING")
print(headline_word_vectors.shape)
headline_word_vectors = Flatten()(headline_word_vectors)
print(headline_word_vectors.shape)

article_input = Input(
    shape=(ARTICLE_LENGTH,),
    dtype='int32',
    name='article_input',
)

article_word_vectors = Embedding(
    input_dim=VOCABULARY_DIM, 
    input_length=ARTICLE_LENGTH,
    output_dim=ARTICLE_EMBEDDING_DIM,
    name='article_embedding'
)(article_input) 

article_word_vectors = Flatten()(article_word_vectors)

layer = word_vectors = concatenate([headline_word_vectors, article_word_vectors])

for dim, activation, dropout in HIDDEN_LAYERS:
    layer = Dense(dim, activation=activation)(layer)
    if dropout:
        layer = Dropout(dropout)(layer)

# generate the predictions. We assume that predicting related correctly will carry over to predicting
# the 'unrelated' stance.
related_prediction = Dense(2, activation='softmax', name='related_prediction')(layer) # Or just use sigmoid...
stance_prediction = Dense(4, activation='softmax', name='stance_prediction')(layer)

model = Model(inputs=[headline_input, article_input], outputs=[related_prediction, stance_prediction])
print(model.summary())
model.compile(**COMPILE)
print(model.summary)

model.fit(
    {
        'headline_input': X_headline,
        'article_input': X_article,
    },
    {
        'related_prediction': y_related,
        'stance_prediction': y_stance,
    },
    validation_data=(
        {
            'headline_input': X_test_headline,
            'article_input': X_test_article,
        },
        {
            'related_prediction': y_test_related,
            'stance_prediction': y_test_stance,
        }
    ),
    **FIT
)

# ===========================================
# VALIDATION 
# ===========================================

pred_related, pred_stance = model.predict([X_test_headline, X_test_article])
report_score([LABELS[np.where(x==1)[0][0]] for x in y_test_stance],
             [LABELS[np.argmax(x)] for x in pred_stance])




