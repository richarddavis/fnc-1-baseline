from configurations.base_config import BaseConfig

class RNNConcatConfig(BaseConfig):
    def __init__(self):
        self.reset()

    def reset(self):
        self.model_module = "rnn_concat"
        self.model_class = "RNNConcat"
        self.vocabulary_dim = 3000
        self.pad_sequences = True
        self.article_length = 1000
        self.headline_length = 100
        self.embedding_dim = 20
        self.optimizer = "nadam"
        self.epochs = 4
        self.batch_size = 64
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
            "compile": {
                'optimizer': self.optimizer,
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy']
            },
            "fit" : {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'verbose': self.verbose
            },
        }
        return rnn_concat
