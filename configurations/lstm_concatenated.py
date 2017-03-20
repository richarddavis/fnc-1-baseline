lstm_concatenated = {
    "model_module": 'lstm_concatenated',
    "model_class": 'LSTMConcatenated',
    "vocabulary_dim": 3000,
    "article_length": 800,
    "headline_length": 100,
    "embedding_dim": 20,
    "hidden_layers": [       
        (50, 'relu', 0.5),
        (50, 'relu', 0.5),
    ],
    "compile": {
        'optimizer': 'nadam', 
        'loss': {
            'related_prediction': 'binary_crossentropy',
            'stance_prediction': 'categorical_crossentropy'
        },
        'loss_weights': {
            'related_prediction': 0.25,
            'stance_prediction': 0.75
        },
        'metrics': ['accuracy']
    },
    "fit" : {
        'epochs': 4,
        'batch_size': 64,
        'verbose': 1
    },
}
