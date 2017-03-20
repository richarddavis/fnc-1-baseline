import random
import os
from collections import defaultdict


def generate_hold_out_split (dataset, training = 0.8, base_dir="splits"):
    r = random.Random()
    # r.seed(1489215)

    article_ids = list(dataset.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # and shuffle that list


    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]

    # write the split body ids out to files for future use
    with open(base_dir+ "/"+ "training_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in training_ids]))

    with open(base_dir+ "/"+ "hold_out_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in hold_out_ids]))

def generate_test_set(dataset, test=0.15, base_dir="splits"):
    "Generates a new test set to hold out permanently"
    r = random.Random()
    article_ids = list(dataset.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # and shuffle that list
    test_ids = article_ids[:int(test * len(article_ids))]
    other_ids = article_ids[int(test * len(article_ids)):]
    print("Generated a test set ({}) and other ({})".format(len(test_ids), len(other_ids)))
    with open(base_dir+ "/"+ "test_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in test_ids]))
    with open(base_dir+ "/"+ "other_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in other_ids]))

def generate_train_val_sets(dataset, val=0.15, base_dir="splits"):
    try: 
        ids = read_ids("other_ids.txt", base_dir)
    except FileNotFoundError:
        raise FileNotFoundError("IDs file {} not found. Run `generate_test_set` first.")
    r = random.Random()
    r.shuffle(ids)
    val_ids = ids[:int(val * len(ids))]
    train_ids = ids[int(val * len(ids)):]
    print("Generated a validation set ({}) and a train set ({})".format(len(val_ids), len(train_ids)))
    with open(base_dir+ "/"+ "val_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in val_ids]))
    with open(base_dir+ "/"+ "train_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in train_ids]))
    
def read_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids

def kfold_split(dataset, training = 0.8, n_folds = 10, base_dir="splits"):
    if not (os.path.exists(base_dir+ "/"+ "training_ids.txt")
            and os.path.exists(base_dir+ "/"+ "hold_out_ids.txt")):
        generate_hold_out_split(dataset,training,base_dir)

    training_ids = read_ids("training_ids.txt", base_dir)
    hold_out_ids = read_ids("hold_out_ids.txt", base_dir)

    folds = []
    for k in range(n_folds):
        folds.append(training_ids[int(k*len(training_ids)/n_folds):int((k+1)*len(training_ids)/n_folds)])

    return folds,hold_out_ids

def get_stances_for_folds(dataset,folds,hold_out):
    stances_folds = defaultdict(list)
    stances_hold_out = []
    for stance in dataset.stances:
        if stance['Body ID'] in hold_out:
            stances_hold_out.append(stance)
        else:
            fold_id = 0
            for fold in folds:
                if stance['Body ID'] in fold:
                    stances_folds[fold_id].append(stance)
                fold_id += 1

    return stances_folds,stances_hold_out
