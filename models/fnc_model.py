from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.callbacks import Callback, History, ModelCheckpoint

class FNCModel:
    """
    Given an instance of FNCConfig (configuration parameters), can build and train a model
    The idea here is that there are discrete steps to model-building: 
    
    1. preprocessing
    2. model definition, which should be able to happen without access to the data
    3. fitting/training

    You should subclass and override methods as necessary. 
    """
    
    def __init__(self, config):
        self.config = config

    def preprocess(self, X_train, y_train, X_test, y_test):
        """
        Maps raw headlines and articles to input ready for this model. It's cleaner to allow the model
        to be built without any data, but part of the model pipeline might depend on data. 
        [[X_headline_train, X_article_train], [X_headline_test, X_article_test]] => input ready for model
        """
        return X_train, y_train, X_test, y_test

    def build_model(self):
        "Builds and returns a compiled model."
        return Model()

    def fit(self, model, X_train, y_train, X_test, y_test):
        "Fits a model with data, then returns the model and the training history"
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_test, y_test),
            callbacks=self.callbacks(),
            **(self.config.get('fit', {}))
        )
        return model, history

    def evaluate(self, model, X_test, y_test):
        return None

    def train(self, X_train, y_train, X_test, y_test):
        "Combines all steps"
        print("=" * 50)
        print("TRAINING {}".format(self.config.slug()))
        print("=" * 50)
        X_train, y_train, X_test, y_test = self.preprocess(X_train, y_train, X_test, y_test)
        model = self.build_model()
        print(model.summary())
        model, history = self.fit(model, X_train, y_train, X_test, y_test)
        self.evaluate(model, X_test, y_test)
        return model, history

    def callbacks(self):
        return [
            History(),
            ModelCheckpoint(self.config.weights_file(), save_best_only=True, save_weights_only=True)
        ]
        
    def tokenize(self, X_train, X_test):
        X_headline_train, X_article_train = X_train
        X_headline_test, X_article_test = X_test
    
        tokenizer = Tokenizer(num_words=self.config['vocabulary_dim'])
        tokenizer.fit_on_texts(X_headline_train + X_article_train)
        X_headline_train = tokenizer.texts_to_sequences(X_headline_train)
        X_article_train = tokenizer.texts_to_sequences(X_article_train)
        X_headline_test = tokenizer.texts_to_sequences(X_headline_test)
        X_article_test = tokenizer.texts_to_sequences(X_article_test)
        if self.config.get('pad_sequences'):
            X_headline_train = pad_sequences(X_headline_train, maxlen=self.config['headline_length'])
            X_article_train = pad_sequences(X_article_train, maxlen=self.config['article_length'])
            X_headline_test = pad_sequences(X_headline_test, maxlen=self.config['headline_length'])
            X_article_test = pad_sequences(X_article_test, maxlen=self.config['article_length'])
        return [[X_headline_train, X_article_train], [X_headline_test, X_article_test]]


        

        
        

