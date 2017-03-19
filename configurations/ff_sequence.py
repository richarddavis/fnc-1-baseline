ff_sequence = {
    "model_module": 'ff_sequence',
    "model_class": 'FFSequence',
    "vocabulary_dim": 3000,
    "pad_sequences": False,
    "matrix_mode": 'binary',
    "article_length": 500,
    "headline_length": 70,
    "article_embedding_dim": 200,
    "headline_embedding_dim": 200,
    "hidden_layers": [       
        (600, 'relu', 0.6)
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
        'epochs': 5,
        'batch_size': 64,
        'verbose': 1
    },
}