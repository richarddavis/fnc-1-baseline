import os
import re
from importlib import import_module
from datetime import datetime
from keras.models import model_from_json
import json
import hashlib

# Should be able to serialize (as a key), save, and look up whether 

# What am I trying to do?

class FNCConfig:
    """
    Represents a FNC configuration
    """

    @classmethod
    def get_all_filenames(cls, results_dir="./results"):
        pattern = "^\w+-(\d{3})-.*\.json$"
        return [os.path.join(results_dir, f) for f in os.listdir(results_dir) if re.search(pattern, f)]

    @classmethod
    def get_all(cls, results_dir="./results"):
        return [cls.load_file(conf) for conf in cls.get_all_filenames(results_dir)]

    @classmethod
    def get_untrained(cls, results_dir="./results"):
        return [conf for conf in cls.get_all() if not conf.is_trained()]

    @classmethod
    def load_file(cls, filename):
        "Given a file path, return a FNCConfig"
        with open(filename, 'r') as source:
            data = json.load(source)
            config = FNCConfig(data['config'])
            config.params = data['params']
            config.history = data['history']
            config.bound_slug = os.path.splitext(os.path.basename(filename))[0]
            return config
        
    def __init__(self, params, results_dir="./results"):
        self.results_dir = results_dir
        self.__dict__.update(params)

    def load(self):
        fingerprint = self.hash()
        matches = list(filter(lambda f: fingerprint in f, os.listdir(self.results_dir)))
        if any(matches):
            return FNCConfig.load_file(matches[0])

    def save(self, model=None, history=None):
        filename = self.config_file()
        with open(filename, 'w') as destination:
            json.dump({
                "config": self.__dict__,
                "params": history.params,
                "history": history.history
            }, destination)
        self.params = history.params
        self.history = history.history
        config.bound_slug = os.path.splitext(os.path.basename(filename))[0]

    def is_trained(self):
        return os.path.exists(self.weights_file())

    def get_model(self):
        try: 
            module = import_module('models.' + self['model_module'])
            model = getattr(module, self['model_class'])
        except: 
            raise ValueError("No such model: {}".format('models.' + self['model_module'] + '.' + self['model_class']))
        return model(self)

    def slug(self):
        if hasattr(self, "bound_slug"):
            return self.bound_slug
        else:
            return "{}-{:03d}-{}".format(self['model_class'], self.next_index(), self.hash())

    def weights_file(self):
        return os.path.join(self.results_dir, "{}.weights".format(self.slug()))

    def config_file(self):
        return os.path.join(self.results_dir, "{}.json".format(self.slug()))

    def next_index(self):
        pattern = "^\w+-(\d{3})-.*\.json$"
        matches = list(filter(lambda f: re.search(pattern, f), os.listdir(self.results_dir)))
        if any(matches):
            get_index = lambda fn: int(re.search(pattern, fn).group(1))
            return max(map(get_index, matches)) + 1
        else: 
            return 0

    def hash(self):
        return hashlib.sha256(json.dumps(self.__dict__).encode('ascii')).hexdigest()[:8]

    def __getitem__(self, attr):
        return self.__dict__[attr]

    def get(self, *args):
        return self.__dict__.get(*args)