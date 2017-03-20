from utils.dataset import DataSet
from utils.generate_data import generate_data, collapse_stances
from utils.generate_test_splits import generate_test_set, generate_train_val_sets, read_ids
from configurations.ff_concat_two_losses import ff_concat_two_losses as conf
from fnc_config import FNCConfig

d = DataSet()
base_dir = "splits"
# generate_test_set(d, test=0.15) # This was already run once; now we will keep using the same IDs.
generate_train_val_sets(d, val=0.2)

train_ids = read_ids("train_ids.txt", base_dir)
X_train_headline, X_train_article, y_train = generate_data(train_ids, d)
val_ids = read_ids("val_ids.txt", base_dir)
X_val_headline, X_val_article, y_val = generate_data(val_ids, d)

for conf in FNCConfig.get_untrained():
    m = conf.get_model()
    model, history = m.train(
        [X_train_headline, X_train_article], 
        y_train,
        [X_val_headline, X_val_article],
        y_val
    )
    conf.save(model, history)

