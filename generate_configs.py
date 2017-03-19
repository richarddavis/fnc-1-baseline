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
# from configurations.ff_concat_two_losses import ff_concat_two_losses as base
# generate_configs(base, {'article_length': [200, 300, 400, 500, 600]})
