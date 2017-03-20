from configurations.base_config import BaseConfig

class RNNConcatConfig(BaseConfig):
    def __init__(self):
        self.reset()

    def reset(self):
        self.model_module = "rnn_concat"
        self.model_class = "RNNConcat"
        self.vocabulary_dim = 3000
        self.pad_sequences = True
        self.article_length = 800
        self.headline_length = 100
        self.embedding_dim = 32
        self.rnn_output_size = 32
        self.bidirectional = True
        self.rnn_depth = 1
        self.optimizer = "nadam"
        self.related_prediction = 'binary_crossentropy'
        self.stance_prediction = 'categorical_crossentropy'
        self.related_prediction_percent = 0
        self.epochs = 10
        self.batch_size = 128
        self.verbose = 1

    def get_config(self):
        rnn_concat = {
            "model_module": self.model_module,
            "model_class": self.model_class,
            "vocabulary_dim": self.vocabulary_dim,
            "pad_sequences": self.pad_sequences,
            "article_length": self.article_length,
            "headline_length": self.headline_length,
            "embedding_dim": self.embedding_dim,
            "rnn_output_size": self.rnn_output_size,
            "bidirectional": self.bidirectional,
            "rnn_depth": self.rnn_depth,
            "compile": {
                'optimizer': self.optimizer,
                'loss': {
                    'related_prediction': self.related_prediction,
                    'stance_prediction': self.stance_prediction,
                },
                'loss_weights': {
                    'related_prediction': self.related_prediction_percent,
                    'stance_prediction': 1 - self.related_prediction_percent,
                },
                'metrics': ['accuracy']
            },
            "fit" : {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'verbose': self.verbose
            },
        }
        return rnn_concat
