import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.layers import Flatten, Merge
from keras.layers.merge import Concatenate
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from utils.dataset import DataSet
from utils.generate_data import generate_data, collapse_stances
from utils.generate_test_splits import generate_hold_out_split, read_ids
from utils.score import report_score, LABELS
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

from models.fnc_model import FNCModel

class FFSequence(FNCModel):

    def preprocess(self, X_train, y_train, X_test, y_test):
        X_train_headline, X_train_article = X_train
        X_test_headline, X_test_article = X_test

        tokenizer = Tokenizer(num_words=self.config['vocabulary_dim'])
        tokenizer.fit_on_texts(X_train_headline + X_train_article)
        X_train_headline = tokenizer.texts_to_sequences(X_train_headline)
        X_train_article = tokenizer.texts_to_sequences(X_train_article)
        X_test_headline = tokenizer.texts_to_sequences(X_test_headline)
        X_test_article = tokenizer.texts_to_sequences(X_test_article)

        X_train_headline = tokenizer.sequences_to_matrix(
            X_train_headline, mode=self.config['matrix_mode'])
        X_train_article = tokenizer.sequences_to_matrix(
            X_train_article, mode=self.config['matrix_mode'])
        X_test_headline = tokenizer.sequences_to_matrix(
            X_test_headline, mode=self.config['matrix_mode'])
        X_test_article = tokenizer.sequences_to_matrix(
            X_test_article, mode=self.config['matrix_mode'])

        y_train_stance = np_utils.to_categorical(y_train)
        y_train_related = np_utils.to_categorical(collapse_stances(y_train))
        y_test_stance = np_utils.to_categorical(y_test)
        y_test_related = np_utils.to_categorical(collapse_stances(y_test))

        return (
            {
                'headline_input': X_train_headline,
                'article_input': X_train_article,
            },
            {
                'related_prediction': y_train_related,
                'stance_prediction': y_train_stance,
            },
            {
                'headline_input': X_test_headline,
                'article_input': X_test_article,
            },
            {
                'related_prediction': y_test_related,
                'stance_prediction': y_test_stance,
            }
        )

    def build_model(self):
        headline_input = Input(
            shape=(self.config['vocabulary_dim'],),
            dtype='float32',
            name='headline_input'
        )
        article_input = Input(
            shape=(self.config['vocabulary_dim'],),
            dtype='float32',
            name='article_input',
        )
        layer = concatenate([headline_input, article_input])

        for dim, activation, dropout in self.config['hidden_layers']:
            layer = Dense(dim, activation=activation)(layer)
            if dropout:
                layer = Dropout(dropout)(layer)

        related_prediction = Dense(2, activation='softmax', name='related_prediction')(layer)
        stance_prediction = Dense(4, activation='softmax', name='stance_prediction')(layer)

        model = Model(inputs=[headline_input, article_input], outputs=[related_prediction, stance_prediction])
        model.compile(**self.config['compile'])
        return model

    def evaluate(self, model, X_test, y_test):
        # This should probably actually be in an evaluate method
        pred_related, pred_stance = model.predict(X_test)
        report_score([LABELS[np.where(x==1)[0][0]] for x in y_test['stance_prediction']],
                [LABELS[np.argmax(x)] for x in pred_stance])
