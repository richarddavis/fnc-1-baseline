from utils.dataset import DataSet
from utils.generate_data import generate_data, collapse_stances
from utils.generate_test_splits import generate_training_test_validation_split, read_ids
from configurations.ff_concat_two_losses import ff_concat_two_losses as conf
from fnc_config import FNCConfig

d = DataSet()
generate_training_test_validation_split(d)
base_dir = "splits"
training_ids = read_ids("training_ids.txt", base_dir)
X_train_headline, X__train_article, y_train = generate_data(training_ids, d)
test_ids = read_ids("test_ids.txt", base_dir)
X_test_headline, X_test_article, y_test = generate_data(test_ids, d)

for conf in FNCConfig.get_untrained():
    m = conf.get_model()
    model, history = m.train(
        [X_train_headline, X__train_article], 
        y_train,
        [X_test_headline, X_test_article],
        y_test
    )
    conf.save(model, history)


