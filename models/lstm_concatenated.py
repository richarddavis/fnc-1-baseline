from models.fnc_model import FNCModel

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Embedding, Flatten, LSTM, GRU, concatenate
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from utils.generate_data import collapse_stances
from utils.score import report_score, LABELS
from keras.utils import np_utils
from keras.utils import plot_model

class LSTMConcatenated(FNCModel):
    "Concatenates headlines and articles, running a LSTM over the whole sequence"

    def preprocess(self, X_train, y_train, X_val, y_val):
        X_train_headline, X_train_article = X_train
        X_val_headline, X_val_article = X_val

        tokenizer = Tokenizer(num_words=self.config['vocabulary_dim'])
        tokenizer.fit_on_texts(X_train_headline + X_train_article)

        X_train_headline = tokenizer.texts_to_sequences(X_train_headline)
        X_train_article = tokenizer.texts_to_sequences(X_train_article)
        X_val_headline = tokenizer.texts_to_sequences(X_val_headline)
        X_val_article = tokenizer.texts_to_sequences(X_val_article)

        X_train_headline = pad_sequences(X_train_headline, maxlen=self.config['headline_length'])
        X_train_article = pad_sequences(X_train_article, maxlen=self.config['article_length'])
        X_val_headline = pad_sequences(X_val_headline, maxlen=self.config['headline_length'])
        X_val_article = pad_sequences(X_val_article, maxlen=self.config['article_length'])
        
        train_separator = np.array([self.config['vocabulary_dim']] * len(X_train_headline))
        val_separator = np.array([self.config['vocabulary_dim']] * len(X_val_headline))

        y_train_stance = np_utils.to_categorical(y_train)
        y_train_related = np_utils.to_categorical(collapse_stances(y_train))
        y_val_stance = np_utils.to_categorical(y_val)
        y_val_related = np_utils.to_categorical(collapse_stances(y_val))

        return (
            [X_train_headline, train_separator, X_train_article],
            [y_train_related, y_train_stance],
            [X_val_headline, val_separator, X_val_article],
            [y_val_related, y_val_stance],
        )

    def build_model(self):
        headline_input = Input(shape=(self.config['headline_length'],), dtype='int32')
        separator_input = Input(shape=(1 ,), dtype='int32')
        article_input = Input(shape=(self.config['article_length'],), dtype='int32')

        concat_input = concatenate([headline_input, separator_input, article_input])

        embed = Embedding(
                input_dim=self.config['vocabulary_dim']+1, 
                output_dim=self.config['embedding_dim'], 
                input_length=self.config['headline_length']  + self.config['article_length'] + 1, 
                mask_zero=True
        )(concat_input)

        layer = gru = GRU(units=128, go_backwards=False)(embed)

        for dim, activation, dropout in self.config['hidden_layers']:
            layer = Dense(dim, activation=activation)(layer)
            if dropout:
                layer = Dropout(dropout)(layer)

        related_prediction = Dense(2, activation='softmax', name='related_prediction')(layer)
        stance_prediction = Dense(4, activation='softmax', name='stance_prediction')(layer)
        model = Model(
            inputs=[headline_input, separator_input, article_input], 
            outputs=[related_prediction, stance_prediction]
        )
        model.compile(**self.config['compile'])
        return model
