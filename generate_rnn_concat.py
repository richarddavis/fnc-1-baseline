from fnc_config import FNCConfig

def generate_configs(base_config, params_to_change, results_dir='./results'):
    "Generates configurations to perform a hyperparameter grid search one dimension at a time."
    for param, values in params_to_change.items():
        for value in values:
            params = {}
            params.update(base)
            params.update({param: value})
            params['config_name'] = "{}: {}".format(param, value)
            FNCConfig(params, results_dir=results_dir).save()

# Example:
if __name__ == '__main__':
    from configurations.rnn_concat import rnn_concat as base
    generate_configs(base, {
        "vocabulary_dim": [3000],
        # "matrix_mode": ["binary", "freq"],
        # "hidden_layers": [
        #     [ (600, 'relu', 0.6), (600, 'relu', 0.6)]
        # ]
    })
