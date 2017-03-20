class BaseConfig:
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def get_config(self):
        pass

    def __getitem__(self, name):
        return getattr(self, name)
    def __setitem__(self, name, value):
        return setattr(self, name, value)
    def __delitem__(self, name):
        return delattr(self, name)
    def __contains__(self, name):
        return hasattr(self, name)
