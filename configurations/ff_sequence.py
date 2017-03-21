from configurations.base_config import BaseConfig

class FF_Sequence_Config(BaseConfig):
    def __init__(self):
        self.reset()

    def reset(self):
        self.model_module = 'ff_sequence'
        self.model_class = 'FFSequence'
        self.vocabulary_dim = 5000
        self.matrix_mode = 'freq'
        self.dense_text = False
        self.dense_dim = 150
        self.hidden_layers = [(600, 'relu', 0.5), (600, 'relu', 0.5), (600, 'relu', 0.5)]
        self.optimizer = 'nadam'
        self.related_prediction = 'binary_crossentropy'
        self.stance_prediction = 'categorical_crossentropy'
        self.related_prediction_percent = 0.25
        self.epochs = 15
        self.batch_size = 64
        self.verbose = 1

    def get_config(self):
        ff_sequence_config = {
            "model_module": self.model_module,
            "model_class": self.model_class,
            "vocabulary_dim": self.vocabulary_dim,
            "matrix_mode": self.matrix_mode,
            "dense_text": self.dense_text,
            "dense_dim": self.dense_dim,
            "hidden_layers": self.hidden_layers,
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
                'verbose': self.verbose,
            },
        }
        return ff_sequence_config
