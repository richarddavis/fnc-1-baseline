ff_concat_two_losses = {
    "model_module": 'ff_concat_two_losses',
    "model_class": 'FFConcatTwoLosses',
    "vocabulary_dim": 300,
    "pad_sequences": True,
    "article_length": 50,
    "headline_length": 10,
    "article_embedding_dim": 20,
    "headline_embedding_dim": 20,
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
        'epochs': 5,
        'batch_size': 64,
        'verbose': 1
    },
}
