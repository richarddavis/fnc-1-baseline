from fnc_config import FNCConfig

def generate_configs(base_config, params_to_change, results_dir='./results'):
    "Generates configurations to perform a hyperparameter grid search one dimension at a time."
    for param, values in params_to_change.items():
        for value in values:
            base.reset()
            base[param] = value
            params = base.get_config()
            params['config_name'] = "{}: {}".format(param, value)
            FNCConfig(params, results_dir=results_dir).save()

# Example:
if __name__ == '__main__':
    from configurations.rnn_concat import RNNConcatConfig
    base = RNNConcatConfig()
    generate_configs(base, {
        "vocabulary_dim": [1000, 3000],
        "bidirectional": [False, True],
        "depth": [1, 2],
        "article_length": [100, 300, 500, 800],
        "related_prediction_percent":[0, 0.25, 0.5, 0.75],
        # "matrix_mode": ["binary", "freq"],
        # "hidden_layers": [
        #     [ (600, 'relu', 0.6), (600, 'relu', 0.6)]
        # ]
    })
