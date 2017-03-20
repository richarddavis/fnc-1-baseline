from models.fnc_model import FNCModel

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Embedding, Flatten, LSTM, GRU, concatenate
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from utils.score import report_score, LABELS
from keras.utils import np_utils
from keras.utils import plot_model

class LSTM(FNCModel):

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
        if self.config.get('share_gru'):
            gru = GRU(units=128, go_backwards=True)
            headline_branch = gru(headline_branch)
            body_branch = gru(body_branch)
        else:
            headline_branch = GRU(units=128, go_backwards=True)(headline_branch)
            body_branch = GRU(units=128, go_backwards=True)(body_branch)

        merged = concatenate([headline_branch, body_branch])
        merged = Dense(400, activation='relu', init='glorot_normal')(merged)
        merged = Dropout(0.2)(merged)
        out = Dense(4, activation='softmax')(merged)
        model = Model(inputs=[headline_input, body_input], output=out)

        # try using different optimizers and different optimizer configs
        model.compile(**self.config['compile'])
        return model
        







