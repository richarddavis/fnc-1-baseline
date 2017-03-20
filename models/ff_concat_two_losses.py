import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.layers import Flatten
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model

from utils.score import report_score, LABELS
from keras.utils import np_utils

from models.fnc_model import FNCModel

class FFConcatTwoLosses(FNCModel):

    def preprocess(self, X_train, y_train, X_val, y_val):
        [X_train_headline, X_train_article], [X_val_headline, X_val_article] = self.tokenize(X_train, X_val)

        y_train_stance = np_utils.to_categorical(y_train)
        y_train_related = np_utils.to_categorical(collapse_stances(y_train))
        y_val_stance = np_utils.to_categorical(y_val)
        y_val_related = np_utils.to_categorical(collapse_stances(y_val))

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
                'headline_input': X_val_headline,
                'article_input': X_val_article,
            },
            {
                'related_prediction': y_val_related,
                'stance_prediction': y_val_stance,
            }
        )

    def build_model(self):
        headline_input = Input(
            shape=(self.config['headline_length'],), 
            dtype='int32', 
            name='headline_input'
        )
        headline_word_vectors = Embedding(
            input_dim=self.config['vocabulary_dim'], 
            input_length=self.config['headline_length'],
            output_dim=self.config['headline_embedding_dim'],
            name='headline_embedding'
        )(headline_input) 
        headline_word_vectors = Flatten()(headline_word_vectors)
        article_input = Input(
            shape=(self.config['article_length'],),
            dtype='int32',
            name='article_input',
        )
        article_word_vectors = Embedding(
            input_dim=self.config['vocabulary_dim'], 
            input_length=self.config['article_length'],
            output_dim=self.config['article_embedding_dim'],
            name='article_embedding'
        )(article_input) 
        article_word_vectors = Flatten()(article_word_vectors)
        layer = word_vectors = concatenate([headline_word_vectors, article_word_vectors])
        for dim, activation, dropout in self.config['hidden_layers']:
            layer = Dense(dim, activation=activation)(layer)
            if dropout:
                layer = Dropout(dropout)(layer)
        related_prediction = Dense(2, activation='softmax', name='related_prediction')(layer)
        stance_prediction = Dense(4, activation='softmax', name='stance_prediction')(layer)
        model = Model(inputs=[headline_input, article_input], outputs=[related_prediction, stance_prediction])
        model.compile(**self.config['compile'])
        return model

    def evaluate(self, model, X_val, y_val):
        # This should probably actually be in an evaluate method
        pred_related, pred_stance = model.predict(X_val)
        report_score([LABELS[np.where(x==1)[0][0]] for x in y_val['stance_prediction']],
                [LABELS[np.argmax(x)] for x in pred_stance])

