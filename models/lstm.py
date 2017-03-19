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
from utils.generate_test_splits import generate_hold_out_split, read_ids
from utils.score import report_score, LABELS
from keras.utils import np_utils
from keras.utils import plot_model

class LSTM(FNCModel):

    def preprocess(self, X_train, y_train, X_test, y_test):
        [X_train_headline, X_train_article], [X_test_headline, X_test_article] = self.tokenize(X_train, X_test)

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        return (
            [X_train_headline, X_train_article],
            y_train,
            [X_test_headline, X_test_article],
            y_test
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
        headline_branch = GRU(output_dim=128, go_backwards=True)(headline_branch)
        body_branch = GRU(output_dim=128, go_backwards=True)(body_branch)

        merged = concatenate([headline_branch, body_branch])
        merged = Dense(400, activation='relu', init='glorot_normal')(merged)
        merged = Dropout(0.2)(merged)
        out = Dense(4, activation='softmax')(merged)
        model = Model(inputs=[headline_input, body_input], output=out)

        # try using different optimizers and different optimizer configs
        model.compile(**self.config['compile'])
        return model
        







