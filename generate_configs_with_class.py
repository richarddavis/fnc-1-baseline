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
    from configurations.ff_sequence_class import FF_Sequence_Config
    base = FF_Sequence_Config()
    generate_configs(base, {
        "matrix_mode": ["binary", "freq"],
        "related_prediction_percent":[0, 0.25, 0.75, 1],
    })
