from fnc_config import FNCConfig
from tabulate import tabulate
import re
import csv
import os
from pprint import pprint

import argparse

def parseNumList(string):
    return sum((_parse(s) for s in string.split(',')), [])

def _parse(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))

parser = argparse.ArgumentParser()
parser.add_argument("--match", '-m', help="String to match test cases against")
parser.add_argument("--dir", help="Directory to search for configs",
        default="./results")
parser.add_argument('--csv', help="Export as a CSV", action="store_true")
parser.add_argument('--sort_metric', '-s', default="val_stance_prediction_acc", 
        help="Metric by which to sort trials")
parser.add_argument('--best_epoch_metric', '-e', default="val_stance_prediction_acc", 
        help="Metric by which to sort trials")
parser.add_argument('--output', '-o', help="CSV output file")
parser.add_argument('--outdir', default="./csv", help="directory for saving output")
parser.add_argument('--range', '-r', type=parseNumList, help="Specify a range of run numbers")
parser.add_argument('--metrics', nargs='*', help="list which metrics to show")
parser.add_argument('--config', action="store_true", help="show differing config options in models")
parser.add_argument('--detail', '-d', action="store_true", help="Show model details")
parser.add_argument('--tableformat', '-t', default="grid", help="Table format (for tabulate)")
args = parser.parse_args()

config_files = FNCConfig.get_all_filenames(results_dir=args.dir)
if args.match:
    config_files = [cf for cf in config_files if re.search(args.match, cf)]
configs = [FNCConfig.load_file(cf, results_dir=args.dir) for cf in config_files]
if args.range:
    configs = [c for c in configs if c.run_number() in args.range]
configs = [conf for conf in configs if conf.is_trained()]

cm = [(c, c.best_epoch_metrics(args.best_epoch_metric) or {}) for c in configs]

def summarize(name):
    return "".join([token[0] for token in name.split("_")])

all_metrics = sorted(list(set(sum([list(m.keys()) for c, m in cm], []))))
if args.metrics:
    metrics = [name for name, summ in zip(all_metrics, (summarize(m) for m in all_metrics)) if summ in args.metrics]
else:
    metrics = all_metrics

def flatten(obj, prefix="", omit=None):
    "returns period-separated paths to all keys in object"
    omit = omit or []
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in omit:
                vals += flatten(v, "{}.{}".format(prefix, k))
    elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if i not in omit:
                    vals += flatten(v, "{}.{}".format(prefix, i))
    else:
        vals.append([prefix, obj])
    return vals

def differs(key, fc):
    return len(set([dict(_fc).get(key) for _fc in fc])) > 1

if args.config:
    flat_configs = [flatten(c.__dict__, omit=['params', 'history', 'bound_slug']) for c in configs]
    for c, fc in zip(configs, flat_configs):
        c._fc = fc
    c_keys = sorted(list(set(sum([[k for k, v in c] for c in flat_configs], [])))) 
    diff_keys = [k for k in c_keys if differs(k, flat_configs) and k != '.config_name']
else:
    diff_keys = ["description"]

headers = ["trial"] + diff_keys + [summarize(m) for m in metrics]
table = []

for c, m in sorted(cm, key=lambda cm_: cm_[1].get(args.sort_metric, -1), reverse=True):
    row = [c.slug()]
    if args.config:
        fc = dict(c._fc)
        for dk in diff_keys:
            row.append(fc.get(dk))
    else:
        row.append(c.get('config_name', "---")[:36])
    for metric in metrics:
        row.append(m.get(metric))
    table.append(row)

def default_out():
    out = args.match or 'all'
    if args.range:
        out += "_{}_to_{}".format(args.range[0], args.range[-1])
    if args.metrics:
        out += "_" + "_".join([summarize(m) for m in args.metrics])
    if args.config:
        out += "_config"
    out += ".csv"
    return out

if args.csv:
    path = os.path.join(args.outdir, args.output or default_out())
    print("Writing to {}".format(path))
    with open(path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        writer.writerows(table)

print(tabulate(table, headers, tablefmt=args.tableformat))

def show_conf(c):
    print("=" * 80)
    print(c.slug(), c.get('config_name', '---'))
    print("=" * 80)
    pprint({ k:v for k, v in c.__dict__.items() if k not in ['params', 'history', 'bound_slug']})
    
    
    cheaders = ["metric"] + [i for i, x in enumerate(c.history.items())]
    ctable = [ [k] + v for k, v in c.history.items() if k in metrics]
    print(tabulate(ctable, cheaders, tablefmt=args.tableformat))

if args.detail:
    for c in configs:
        show_conf(c)
    
