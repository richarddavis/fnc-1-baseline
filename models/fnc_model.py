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

    def preprocess(self, X_train, y_train, X_val, y_val):
        """
        Maps raw headlines and articles to input ready for this model. It's cleaner to allow the model
        to be built without any data, but part of the model pipeline might depend on data. 
        [[X_train_headline, X_train_article], [X_val_headline, X_val_article]] => input ready for model
        """
        return X_train, y_train, X_val, y_val

    def build_model(self):
        "Builds and returns a compiled model."
        return Model()

    def fit(self, model, X_train, y_train, X_val, y_val):
        "Fits a model with data, then returns the model and the training history"
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            callbacks=self.callbacks(),
            **(self.config.get('fit', {}))
        )
        return model, history

    def evaluate(self, model, X_val, y_val):
        return None

    def train(self, X_train, y_train, X_val, y_val):
        "Combines all steps"
        print("=" * 80)
        print("TRAINING {} ({})".format(self.config.slug(), self.config.get("config_name", "[no name]")))
        print("=" * 80)
        X_train, y_train, X_val, y_val = self.preprocess(X_train, y_train, X_val, y_val)
        model = self.build_model()
        print(model.summary())
        model, history = self.fit(model, X_train, y_train, X_val, y_val)
        self.evaluate(model, X_val, y_val)
        return model, history

    def callbacks(self):
        return [
            History(),
            ModelCheckpoint(self.config.weights_file(), save_best_only=True, save_weights_only=True)
        ]
        
    def tokenize(self, X_train, X_val):
        X_train_headline, X_train_article = X_train
        X_val_headline, X_val_article = X_val
    
        tokenizer = Tokenizer(num_words=self.config['vocabulary_dim'])
        tokenizer.fit_on_texts(X_train_headline + X_train_article)
        X_train_headline = tokenizer.texts_to_sequences(X_train_headline)
        X_train_article = tokenizer.texts_to_sequences(X_train_article)
        X_val_headline = tokenizer.texts_to_sequences(X_val_headline)
        X_val_article = tokenizer.texts_to_sequences(X_val_article)
        if self.config.get('pad_sequences'):
            X_train_headline = pad_sequences(X_train_headline, maxlen=self.config['headline_length'])
            X_train_article = pad_sequences(X_train_article, maxlen=self.config['article_length'])
            X_val_headline = pad_sequences(X_val_headline, maxlen=self.config['headline_length'])
            X_val_article = pad_sequences(X_val_article, maxlen=self.config['article_length'])
        return [[X_train_headline, X_train_article], [X_val_headline, X_val_article]]


        

        
        

