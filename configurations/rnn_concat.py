rnn_concat = {
    "model_module": 'rnn_concat',
    "model_class": 'RNNConcat',
    "vocabulary_dim": 3000,
    "pad_sequences": True,
    "article_length": 1000,
    "headline_length": 100,
    "embedding_dim": 20,
    "compile": {
        'optimizer': 'nadam',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    },
    "fit" : {
        'epochs': 4,
        'batch_size': 64,
        'verbose': 1
    },
}
