# Training Models

The general strategy for training models is to specify a model architecture by subclassing 
`models.fnc_model.FNCModel` and then creating instances of `fnc_config.FNCConfig` specifying
hyperparameters for the models. Invoking `save` on a FNCConfig instance will write the 
configuration to a file (also creating a weights file if the model has been trained). 

## Example

    # Create a bunch of configs. 
    from generate_configs import generate_configs
    from configurations.ff_concat_two_losses import ff_concat_two_losses as base 
    generate_configs(base, {'article_length': [200, 300, 400, 500, 600]})

    # Now call `python train_all.py`. This short script goes through all the configs
    # in its specified directory, and trains the configs which have not yet been trained.
    
    # Finally, we can see how things went
    FNCConfig.show_all()

## Details

There are a few expected config keys:

- `config_name` is a description of this test case. 
- `model_module` names the module containing this config's model, in ./models
- `model_class` names the model to use (a subclass of FNCModel)

FNCModel is a wrapper over a functional Keras model. See more comments inside; the general idea
is that there are several phases in a model's architecture:

- `preprocess` transforms the raw data (training and test) into a form suitable for consumption
  by the model. Feature extraction happens here. 
- `get_model` builds the actual keras model without accessing the data
- `fit` fits the model to the data
- `evaluate` currently just prints out the confusion matrix
- `train` is a one-stop shop, performing everything

## Working in parallel
This arrangement is well-suited to farming out the work. If we create the configurations we want to 
test from one central location, we can put them in different directories and check them in to the repo.
Then, each machine can run `train_all.py`

