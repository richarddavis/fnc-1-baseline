class FF_Sequence_Config():
    def __init__(self):
        self.reset()

    def reset(self):
        self.model_module = 'ff_sequence'
        self.model_class = 'FFSequence'
        self.vocabulary_dim = 3000
        self.matrix_mode = 'binary'
        self.hidden_layers = [(600, 'relu', 0.6)]
        self.optimizer = 'nadam'
        self.related_prediction = 'binary_crossentropy'
        self.stance_prediction = 'categorical_crossentropy'
        self.related_prediction_percent = 0
        self.epochs = 4
        self.batch_size = 64
        self.verbose = 1

    def get_config(self):
        ff_sequence_config = {
            "model_module": self.model_module,
            "model_class": self.model_class,
            "vocabulary_dim": self.vocabulary_dim,
            "matrix_mode": self.matrix_mode,
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

    def __getitem__(self, name):
        return getattr(self, name)
    def __setitem__(self, name, value):
        return setattr(self, name, value)
    def __delitem__(self, name):
        return delattr(self, name)
    def __contains__(self, name):
        return hasattr(self, name)
