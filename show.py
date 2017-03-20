from fnc_config import FNCConfig
from tabulate import tabulate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--match", '-m', help="String to match test cases against")
parser.add_argument("--dir", '-d', help="Directory to search for configs",
        default="./results")
parser.add_argument('--csv', help="Export as a CSV", action="store_true")
parser.add_argument('--sort_metric', '-s', default="val_stance_prediction_acc", 
        help="Metric by which to sort trials")
parser.add_argument('--best_epoch_metric', '-e', default="val_stance_prediction_acc", 
        help="Metric by which to sort trials")
parser.add_argument('--output', '-o', help="CSV output file")
args = parser.parse_args()

config_files = FNCConfig.get_all_filenames(results_dir=args.dir)
if args.match:
    config_files = [cf for cf in config_files if re.search(args.match, cf)]
configs = [FNCConfig.load_file(cf, results_dir=args.dir) for cf in config_files]
configs = [conf for conf in configs if conf.is_trained()]

cm = [(c, c.best_epoch_metrics(args.best_epoch_metric) or {}) for c in configs]

metrics = sorted(list(set(sum([list(m.keys()) for c, m in cm], []))))

def summarize(name):
    return "".join([token[0] for token in name.split("_")])

headers = ["trial", "description"] + [summarize(m) for m in metrics]
table = []

for c, m in sorted(cm, key=lambda cm_: cm_[1].get(args.sort_metric, -1), reverse=True):
    row = [c.slug(), c.get('config_name', "---")[:24]]
    for metric in metrics:
        row.append(m.get(metric))
    table.append(row)

print(tabulate(table, headers, tablefmt="grid"))
    
