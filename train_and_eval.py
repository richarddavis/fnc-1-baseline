from utils.dataset import DataSet
from utils.generate_data import generate_data, collapse_stances
from utils.generate_test_splits import generate_test_set, generate_train_val_sets, read_ids
from configurations.ff_concat_two_losses import ff_concat_two_losses as conf
from fnc_config import FNCConfig
import re
import argparse
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("--match", '-m', help="String to match test cases against")
parser.add_argument("--dir", '-d', help="Directory to search for configs",
        default="./results")
args = parser.parse_args()

d = DataSet()
base_dir = "splits"
generate_train_val_sets(d, val=0.2)

train_ids = read_ids("train_ids.txt", base_dir)
X_train_headline, X_train_article, y_train = generate_data(train_ids, d)
val_ids = read_ids("val_ids.txt", base_dir)
X_val_headline, X_val_article, y_val = generate_data(val_ids, d)
test_ids = read_ids("test_ids.txt", base_dir)
X_test_headline, X_test_article, y_test = generate_data(test_ids, d)

config_files = FNCConfig.get_all_filenames(results_dir=args.dir)

if args.match:
    config_files = [cf for cf in config_files if re.search(args.match, cf)]

configs = [FNCConfig.load_file(cf, results_dir=args.dir) for cf in config_files]

configs = sorted(configs, key=lambda c: c.run_number())

models = []

for conf in configs:
    m = conf.get_model()
    model, history = m.train(
        [X_train_headline + X_val_headline, X_train_article + X_val_article], 
        y_train + y_val,
        [X_val_headline, X_val_article],
        y_val
    )
    results = model.evaluate([X_test_headline, X_test_article], y_test, batch_size=64)
    models.append(model)
    print(tabulate([results], model.metrics_names))
