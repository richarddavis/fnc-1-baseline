from models.fnc_model import FNCModel

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
from utils.score import report_score, LABELS
from keras.utils import np_utils
from keras.utils import plot_model

class RNNConcat(FNCModel):

    def preprocess(self, X_train, y_train, X_val, y_val):
        [X_train_headline, X_train_article], [X_val_headline, X_val_article] = self.tokenize(X_train, X_val)

        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_val)

        return (
            [X_train_headline, X_train_article],
            y_train,
            [X_val_headline, X_val_article],
            y_val
        )

    def tokenize(self, X_train, X_val):
        # X_train and X_val are each lists of [headline, article]
        X_headline_train, X_article_train = X_train
        X_headline_val, X_article_val = X_val

        start_headline_token = '<HEADLINE>'
        start_article_token = '<ARTICLE>'
        X_headline_train = [" ".join([start_headline_token, headline]) for headline in X_headline_train]
        X_article_train = [" ".join([start_article_token, article]) for article in X_article_train]
        X_headline_val = [" ".join([start_headline_token, headline]) for headline in X_headline_val]
        X_article_val = [" ".join([start_article_token, article]) for article in X_article_val]

        # Add check that start_headline_token and start_article_token make it

        tokenizer = Tokenizer(num_words=self.config['vocabulary_dim'])
        tokenizer.fit_on_texts(X_headline_train + X_article_train + X_headline_val + X_article_val)
        X_headline_train = tokenizer.texts_to_sequences(X_headline_train)
        X_article_train = tokenizer.texts_to_sequences(X_article_train)
        X_headline_val = tokenizer.texts_to_sequences(X_headline_val)
        X_article_val = tokenizer.texts_to_sequences(X_article_val)
        if self.config.get('pad_sequences'):
            X_headline_train = pad_sequences(X_headline_train, maxlen=self.config['headline_length'])
            X_article_train = pad_sequences(X_article_train, maxlen=self.config['article_length'])
            X_headline_val = pad_sequences(X_headline_val, maxlen=self.config['headline_length'])
            X_article_val = pad_sequences(X_article_val, maxlen=self.config['article_length'])
        return [[X_headline_train, X_article_train], [X_headline_val, X_article_val]]

    def build_model(self):

        headline_input = Input(shape=(self.config['headline_length'],), dtype='int32')
        body_input = Input(shape=(self.config['article_length'],), dtype='int32')
        headline_branch = Embedding(
                input_dim=self.config['vocabulary_dim']+2,
                output_dim=self.config['embedding_dim'],
                input_length=self.config['headline_length'],
                mask_zero=True
        )(headline_input)
        body_branch = Embedding(
                input_dim=self.config['vocabulary_dim']+2,
                output_dim=self.config['embedding_dim'],
                input_length=self.config['article_length'],
                mask_zero=True
        )(body_input)

        merged = concatenate([headline_branch, body_branch], axis=1)
        merged_length = self.config['article_length'] + self.config['headline_length']

        merged = GRU(output_dim=128, go_backwards=False)(merged)

        # merged = Dense(400, activation='relu', init='glorot_normal')(merged)
        # merged = Dropout(0.2)(merged)
        out = Dense(4, activation='softmax')(merged)
        model = Model(inputs=[headline_input, body_input], output=out)



        # try using different optimizers and different optimizer configs
        model.compile(**self.config['compile'])
        return model
